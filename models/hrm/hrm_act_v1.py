from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding
from models.layers import CastedEmbedding, CastedLinear

class MurderTreeV2Config(BaseModel):
    hidden_size: int = 1024
    vocab_size: int = 32768
    seq_len: int = 2048
    num_heads: int = 16
    rope_theta: float = 100_000.0

    tier1_depth: int = 2
    tier1_count: int = 32
    tier2_depth: int = 4
    tier2_count: int = 8
    tier3_depth: int = 8
    tier3_count: int = 3

    verifier_depth: int = 32
    slow_verify_every: int = 8
    max_steps: int = 64
    ponder_cost_weight: float = 0.01
    evolution_trigger_threshold: float = 0.35
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

    def forward(self, h: torch.Tensor, inject: Optional[torch.Tensor] = None):
        x = self.norm1(h)
        if inject is not None:
            x = x + inject
        x = rms_norm(x + self.attn(hidden_states=x), 1e-5)
        x = rms_norm(x + self.gate.tanh() * self.mlp(self.norm2(x)), 1e-5)
        return x


class TieredProposerBank(nn.Module):
    def __init__(self, dim: int, depth: int, count: int):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(*(GriffinBlock(dim) for _ in range(depth)))
            for _ in range(count)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, states: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
        # states: [count, B, dim], inject: [B, dim]
        outs = []
        for i, net in enumerate(self.nets):
            h = states[i] if i < states.shape[0] else states[0]
            h = net(h)
            outs.append(self.norm(h + inject))
        return torch.stack(outs, dim=0)  # [count, B, dim]


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor        # [total_experts, B, D]
    z_L: torch.Tensor        # low-level memory, unused here but kept for compatibility
    z_mem: Optional[torch.Tensor] = None      # [num_policy_slots, B, D]
    evolution_mask: Optional[torch.Tensor] = None  # [num_policy_slots]


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class MurderTreeV2Core(nn.Module):
    def __init__(self, cfg: MurderTreeV2Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        self.dtype = getattr(torch, cfg.forward_dtype)

        self.embed = CastedEmbedding(cfg.vocab_size, d, cast_to=self.dtype)
        self.rotary = RotaryEmbedding(dim=d // cfg.num_heads, base=cfg.rope_theta)

        self.input_proj = nn.Linear(d, d, bias=False)

        self.tier1 = TieredProposerBank(d, cfg.tier1_depth, cfg.tier1_count)
        self.tier2 = TieredProposerBank(d, cfg.tier2_depth, cfg.tier2_count)
        self.tier3 = TieredProposerBank(d, cfg.tier3_depth, cfg.tier3_count)

        self.verifier = nn.ModuleList([GriffinBlock(d) for _ in range(cfg.verifier_depth)])
        self.fast_critic = nn.Linear(d, 1, bias=False)
        self.slow_critic = nn.Linear(d, 1, bias=False)


        self.lm_head = CastedLinear(d, cfg.vocab_size, bias=False)
        self.halt_head = nn.Linear(d, 2, bias=True)
        self.evolve_head = nn.Linear(d, 1, bias=False)

        self.policy_memory = nn.Parameter(torch.zeros(cfg.num_policy_slots, d))
        nn.init.trunc_normal_(self.policy_memory, std=0.02)
        self.policy_router = nn.Linear(d, cfg.num_policy_slots)

        self.recurrent_gate = nn.Linear(d, d, bias=False)

        self.apply(self._init_weights)
        with torch.no_grad():
            self.halt_head.bias.fill_(-6.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        B, T = input_ids.shape
        d = self.cfg.hidden_size
        device = input_ids.device

        # === Embed + project + pool over sequence length ===
        pos = torch.arange(T, device=device)
        x = self.embed(input_ids.to(torch.int32)) * math.sqrt(d)
        x = x + self.rotary(pos)
        x = self.input_proj(x)               # [B, T, D]
        inject = x.mean(dim=1)                       # [B, D]  ← this is what we inject

        # === Collapse previous expert states to current recurrent state ===
        h = carry.z_H.mean(dim=0)                    # [B, D]

        # === Inject discovered policies ===
        if carry.evolution_mask is not None and carry.evolution_mask.any():
            active_policies = carry.z_mem[carry.evolution_mask]  # [k, B, D]
            h = h + active_policies.mean(0) * 0.3


        t1_states = carry.z_H[:self.cfg.tier1_count]
        t1_out = self.tier1(t1_states, inject)

        t2_states = carry.z_H[self.cfg.tier1_count:self.cfg.tier1_count + self.cfg.tier2_count]
        t2_out = self.tier2(t2_states, inject)

        t3_states = carry.z_H[-self.cfg.tier3_count:]
        candidates = self.tier3(t3_states, inject)           # [tier3_count, B, D]


        verify_in = candidates.mean(0)                       # [B, D]
        slow = (carry.steps % self.cfg.slow_verify_every == 0).any()
        if slow and self.training:
            for block in self.verifier:
                verify_in = block(verify_in)
            value = self.slow_critic(verify_in.mean(0, keepdim=True))
        else:
            value = self.fast_critic(verify_in)

        # === Recurrent update ===
        gate = torch.sigmoid(self.recurrent_gate(h))
        new_h = rms_norm(h + gate * verify_in, self.cfg.rms_eps)   # [B, D]


        logits = self.lm_head(new_h)                                 # [B, V]
        q = self.halt_head(new_h)
        q_halt, q_cont = q[:, 0], q[:, 1]
        q_evolve = self.evolve_head(new_h.mean(0, keepdim=True))

        # === Self-evolution (training only) ===
        new_policy = None
        evolve_now = (q_evolve.sigmoid() > self.cfg.evolution_trigger_threshold) and self.training
        if evolve_now:
            noise = torch.randn(self.cfg.evolution_candidates, B, d, device=device, dtype=self.dtype) * 0.02
            cands = new_h.unsqueeze(0) + noise
            cands = rms_norm(cands, self.cfg.rms_eps)
            scores = self.slow_critic(cands.mean(1))          # [candidates, 1]
            best = scores.squeeze(1).argmax(0)                # [B]
            new_policy = cands[best, torch.arange(B)]

            if carry.z_mem is None:
                carry.z_mem = torch.zeros(self.cfg.num_policy_slots, B, d, device=device, dtype=self.dtype)
                carry.evolution_mask = torch.zeros(self.cfg.num_policy_slots, dtype=torch.bool, device=device)

            mask = ~carry.evolution_mask
            if mask.any():
                slot = mask.nonzero(as_tuple=True)[0][0]      # first free slot (shared across batch)
                carry.z_mem[slot] = new_policy.detach()
                carry.evolution_mask[slot] = True

        return new_h, logits, (q_halt, q_cont), new_policy



class HierarchicalReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.old_config = config_dict
        self.cfg = MurderTreeV2Config(
            hidden_size=config_dict.get("hidden_size", 1024),
            vocab_size=config_dict.get("vocab_size", 32768),
            forward_dtype=config_dict.get("forward_dtype", "bfloat16"),
        )
        self.core = MurderTreeV2Core(self.cfg)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        bs = batch["inputs"].shape[0]
        d = self.cfg.hidden_size
        dtype = self.core.dtype
        device = next(self.core.parameters()).device

        total_experts = self.cfg.tier1_count + self.cfg.tier2_count + self.cfg.tier3_count
        fake_H = torch.zeros(total_experts, bs, d, device=device, dtype=dtype)

        inner = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=fake_H,
            z_L=torch.zeros(bs, batch["inputs"].shape[1] + 64, d, device=device,|False dtype=dtype),
            z_mem=None,
            evolution_mask=None,
        )
        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=inner,
            steps=torch.zeros(bs, dtype=torch.int32, device=device),
            halted=torch.zeros(bs, dtype=torch.bool, device=device),
            current_data={k: v.clone() for k, v in batch.items()},
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
        inner = carry.inner_carry
        inner.steps = carry.steps  # expose for slow verifier logic

        new_h, logits, (q_halt, q_cont), _ = self.core(inner, batch["inputs"])

        # Update all expert states with the new recurrent state (MoE-style broadcast)
        total_experts = self.cfg.tier1_count + self.cfg.tier2_count + self.cfg.tier3_count
        new_z_H = inner.z_H.clone()
        new_z_H[:] = new_h.unsqueeze(0)

        new_inner = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=new_z_H.detach(),
            z_L=inner.z_L,
            z_mem=inner.z_mem,
            evolution_mask=inner.evolution_mask,
        )
        carry.inner_carry = new_inner
        carry.steps = carry.steps + 1

        with torch.no_grad():
            should_halt = (carry.steps >= self.cfg.max_steps) | (q_halt > q_cont)
            if self.training:
                should_halt = should_halt | (torch.rand_like(q_cont) < 0.05)
            carry.halted = carry.halted | should_halt

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_cont,
            "loss_aux": self.cfg.ponder_cost_weight * carry.steps.float().mean() if self.training else None,
            "active_policies": inner.evolution_mask.sum().item() if inner.evolution_mask is not None else 0,
        }
        return carry, outputs
