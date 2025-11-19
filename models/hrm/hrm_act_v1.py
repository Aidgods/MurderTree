

from typing import Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from dataclasses import dataclass

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding
from models.layers import CastedEmbedding, CastedLinear


class MurderTreeV2Config(BaseModel):
    hidden_size: int = 1024
    vocab_size: int = 32768
    num_heads: int = 16
    rope_theta: float = 100_000.0

    # Tiered proposers (progressively deeper & wider)
    tier1_dim: int = 512
    tier1_depth: int = 3
    tier1_count: int = 32

    tier2_dim: int = 768
    tier2_depth: int = 5
    tier2_count: int = 12

    tier3_dim: int = 1024
    tier3_depth: int = 8
    tier3_count: int = 6

    # Verification & evolution
    verifier_depth: int = 24
    slow_verify_every: int = 8
    max_steps: int = 96
    ponder_cost_weight: float = 0.01
    evolution_trigger_threshold: float = 0.30
    num_policy_slots: int = 12
    evolution_candidates: int = 8
    forward_dtype: str = "bfloat16"
    rms_eps: float = 1e-5

class GriffinBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(hidden_size=dim, num_heads=heads, causal=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, expansion=4.0)
        self.gate = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor, inject: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        if inject is not None:
            h = h + inject
        h = rms_norm(h + self.attn(h), 1e-5)
        h = rms_norm(h + self.gate.tanh() * self.mlp(self.norm2(h)), 1e-5)
        return h


class TieredProposerBank(nn.Module):
    def __init__(self, dim: int, depth: int, count: int):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(*(GriffinBlock(dim, heads=max(8, dim // 64)) for _ in range(depth)))
            for _ in range(count)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward_all(self, states: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
        # states: [count, B, dim], inject: [B, dim] → returns [count, B, dim]
        outs = []
        for i, net in enumerate(self.nets):
            h = net(states[i] + inject)
            outs.append(self.norm(h))
        return torch.stack(outs, dim=0)


@dataclass
class Carry:
    z_global: torch.Tensor      # [B, D] main recurrent state
    z_mem: torch.Tensor         # [num_slots, B, D] discovered policies
    steps: torch.Tensor         # [B]
    halted: torch.Tensor        # [B]
    evolution_mask: torch.Tensor  # [num_slots]

class MurderTreeV2(nn.Module):
    def __init__(self, config: MurderTreeV2Config):
        super().__init__()
        self.cfg = config
        d = config.hidden_size
        self.dtype = getattr(torch, config.forward_dtype)

        self.embed = CastedEmbedding(config.vocab_size, d, cast_to=self.dtype)
        self.rotary = RotaryEmbedding(dim=d // config.num_heads, base=config.rope_theta)

        # Tiered proposers
        self.tier1 = TieredProposerBank(config.tier1_dim, config.tier1_depth, config.tier1_count)
        self.tier2 = TieredProposerBank(config.tier2_dim, config.tier2_depth, config.tier2_count)
        self.tier3 = TieredProposerBank(config.tier3_dim, config.tier3_depth, config.tier3_count)

        self.up1 = SwiGLU(config.tier1_dim, config.tier2_dim)
        self.up2 = SwiGLU(config.tier2_dim, config.tier3_dim)
        self.tier3_to_hidden = nn.Identity()  # in case tier3_dim != hidden_size

        # Verifier & critics
        self.verifier = nn.ModuleList([GriffinBlock(d) for _ in range(config.verifier_depth)])
        self.fast_critic = nn.Linear(d, 1, bias=False)
        self.slow_critic = nn.Linear(d, 1, bias=False)

        # Heads
        self.lm_head = CastedLinear(d, config.vocab_size, bias=False)
        self.halt_head = nn.Linear(d, 2, bias=True)
        self.evolve_head = nn.Linear(d, 1)

        # Policy memory
        self.policy_memory = nn.Parameter(torch.zeros(config.num_policy_slots, d))
        nn.init.trunc_normal_(self.policy_memory, std=0.02)

        # Gating
        self.recurrent_gate = nn.Linear(d, d, bias=False)

        self.apply(self._init_weights)
        with torch.no_grad():
            self.halt_head.bias.fill_(-6.0)  # continue early

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def initial_carry(self, batch_size: int, device: torch.device) -> Carry:
        d = self.cfg.hidden_size
        return Carry(
            z_global=torch.zeros(batch_size, d, device=device, dtype=self.dtype),
            z_mem=torch.zeros(self.cfg.num_policy_slots, batch_size, d, device=device, dtype=self.dtype),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.zeros(batch_size, dtype=torch.bool, device=device),
            evolution_mask=torch.zeros(self.cfg.num_policy_slots, dtype=torch.bool, device=device),
        )

    def forward(self, carry: Carry, input_ids: torch.Tensor) -> Tuple[Carry, Dict]:
        B, T = input_ids.shape
        d = self.cfg.hidden_size
        device = input_ids.device
        dtype = self.dtype

        # === Embedding + inject ===
        x = self.embed(input_ids) * math.sqrt(d)
        pos = torch.arange(T, device=device)
        x = x + self.rotary(pos)
        inject = x.mean(dim=1)  # [B, D]

        h = carry.z_global

        # === Inject discovered policies ===
        if carry.evolution_mask.any():
            active = carry.z_mem[carry.evolution_mask]
            h = h + active.mean(0) * 0.3

        # Sequential Hierarchical Top-K Proposing (the magic)
        base_inject = h

        # Tier 1 — many shallow experts
        t1_states = torch.zeros(self.cfg.tier1_count, B, self.cfg.tier1_dim, device=device, dtype=dtype)
        t1_inject = base_inject.unsqueeze(0).expand(self.cfg.tier1_count, -1, -1)
        t1_candidates = self.tier1.forward_all(t1_states, t1_inject)  # [32, B, 512]
        t1_scores = self.fast_critic(t1_candidates.mean(-1)).squeeze(-1)  # [32, B]
        topk_t1, idx1 = torch.topk(t1_scores, k=8, dim=0)  # [8, B]
        selected_t1 = t1_candidates.gather(0, idx1.unsqueeze(-1).expand(-1, -1, self.cfg.tier1_dim))

        # Tier 2 — medium depth, attends to best Tier 1
        t2_inject = self.up1(selected_t1.mean(0)) + base_inject
        t2_states = torch.zeros(self.cfg.tier2_count, B, self.cfg.tier2_dim, device=device, dtype=dtype)
        t2_inject = t2_inject.unsqueeze(0).expand(self.cfg.tier2_count, -1, -1)
        t2_candidates = self.tier2.forward_all(t2_states, t2_inject)  # [12, B, 768]
        t2_scores = self.fast_critic(t2_candidates.mean(-1)).squeeze(-1)
        topk_t2, idx2 = torch.topk(t2_scores, k=4, dim=0)
        selected_t2 = t2_candidates.gather(0, idx2.unsqueeze(-1).expand(-1, -1, self.cfg.tier2_dim))

        # Tier 3 — few deep experts, final proposals
        t3_inject = self.up2(selected_t2.mean(0)) + base_inject
        t3_states = torch.zeros(self.cfg.tier3_count, B, d, device=device, dtype=dtype)
        t3_inject = t3_inject.unsqueeze(0).expand(self.cfg.tier3_count, -1, -1)
        t3_candidates = self.tier3.forward_all(t3_states, t3_inject)  # [6, B, 1024]
        t3_candidates = self.tier3_to_hidden(t3_candidates)

        # Final selection
        candidate_scores = self.fast_critic(t3_candidates.mean(-1)).squeeze(-1)  # [6, B]
        best_idx = candidate_scores.argmax(0)  # [B]
        verify_in = t3_candidates[best_idx, torch.arange(B)]

        # Verification (slow every N steps)
        slow = (carry.steps[0] % self.cfg.slow_verify_every == 0)
        if slow and self.training:
            for block in self.verifier:
                verify_in = block(verify_in, inject)
            value = self.slow_critic(verify_in)
        else:
            value = self.fast_critic(verify_in)

        # Recurrent update
        gate = torch.sigmoid(self.recurrent_gate(h))
        new_h = rms_norm(h + gate * verify_in, self.cfg.rms_eps)

        logits = self.lm_head(new_h)
        q = self.halt_head(new_h)
        q_halt, q_cont = q[:, 0], q[:, 1]
        q_evolve = self.evolve_head(new_h.mean(0, keepdim=True))

        # ===================================================================
        # Self-Evolution (training only)
        # ===================================================================
        new_policy = None
        evolve_now = (q_evolve.sigmoid() > self.cfg.evolution_trigger_threshold) & self.training
        if evolve_now.any():
            noise = torch.randn(self.cfg.evolution_candidates, B, d, device=device, dtype=dtype) * 0.02
            cands = new_h.unsqueeze(0) + noise
            cands = rms_norm(cands, self.cfg.rms_eps)
            scores = self.slow_critic(cands.mean(1))
            best = scores.squeeze(1).argmax(0)
            new_policy = cands[best, torch.arange(B)]

            mask = ~carry.evolution_mask
            if mask.any():
                slot = mask.nonzero(as_tuple=False)[0, 0]  # first free slot
                carry.z_mem[slot] = new_policy.detach()
                carry.evolution_mask[slot] = True

        should_halt = (carry.steps >= self.cfg.max_steps) | (q_halt > q_cont)
        if self.training:
            should_halt = should_halt | (torch.rand_like(q_cont) < 0.05)

        new_carry = Carry(
            z_global=new_h.detach(),
            z_mem=carry.z_mem.detach(),
            steps=carry.steps + 1,
            halted=carry.halted | should_halt,
            evolution_mask=carry.evolution_mask.clone(),
        )

        outputs = {
            "logits": logits,
            "q_halt": q_halt,
            "q_continue": q_cont,
            "new_reasoning_module": new_policy,
            "active_policies": carry.evolution_mask.sum().item(),
        }
        if self.training:
            outputs["loss_aux"] = self.cfg.ponder_cost_weight * carry.steps.float().mean()

        return new_carry, outputs
