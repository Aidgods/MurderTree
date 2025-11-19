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


class MurderTreeV3Config(BaseModel):
    hidden_size: int = 1024
    vocab_size: int = 32768
    seq_len: int = 2048
    num_heads: int = 16
    rope_theta: float = 100_000.0

    # Recursive tiers
    t1_dim: int = 512
    t1_steps: int = 14
    t2_dim: int = 768
    t2_steps: int = 12
    t3_dim: int = 1024
    t3_steps: int = 10

    verifier_depth: int = 16
    slow_verify_every: int = 8

    max_ponder_steps: int = 96
    ponder_cost_weight: float = 0.01

    # Self-evolution
    num_policy_slots: int = 16
    evolution_threshold: float = 0.33
    evolution_candidates: int = 12
    evolution_noise: float = 0.025

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


class RecursiveTier(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.blocks = nn.ModuleList([GriffinBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, inject: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            z = z + self.blocks[0](self.norm(z + inject))   # shared block is fine & faster
        return z


@dataclass
class Carry:
    z: torch.Tensor                 # [B, D] global recurrent state
    policies: torch.Tensor          # [num_slots, B, D]
    step: torch.Tensor              # [B] long
    halted: torch.Tensor            # [B] bool
    policy_mask: torch.Tensor       # [num_slots] bool


class MurderTreeV3(nn.Module):
    def __init__(self, cfg: MurderTreeV3Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        self.dtype = getattr(torch, cfg.forward_dtype)

        self.embed = CastedEmbedding(cfg.vocab_size, d, cast_to=self.dtype)
        self.rotary = RotaryEmbedding(dim=d // cfg.num_heads, base=cfg.rope_theta)

        # Recursive tiers (best of the recursive version)
        self.tier1 = RecursiveTier(cfg.t1_dim, depth=3, heads=8)
        self.tier2 = RecursiveTier(cfg.t2_dim, depth=4, heads=12)
        self.tier3 = RecursiveTier(cfg.t3_dim, depth=5, heads=16)

        self.up1 = SwiGLU(cfg.t1_dim, cfg.t2_dim)
        self.up2 = SwiGLU(cfg.t2_dim, cfg.t3_dim)
        self.down3 = SwiGLU(cfg.t3_dim, d) if cfg.t3_dim != d else nn.Identity()

        # Verification
        self.verifier = nn.ModuleList([GriffinBlock(d, heads=16) for _ in range(cfg.verifier_depth)])
        self.fast_critic = nn.Linear(d, 1, bias=False)
        self.slow_critic = nn.Linear(d, 1, bias=False)

        self.lm_head = CastedLinear(d, cfg.vocab_size, bias=False)
        self.halt_head = nn.Linear(d, 2)
        self.evolve_head = nn.Linear(d, 1)

        # Learned policy memory + per-slot gating (new & better)
        self.policy_memory = nn.Parameter(torch.zeros(cfg.num_policy_slots, d))
        nn.init.trunc_normal_(self.policy_memory, std=0.02)
        self.policy_gate = nn.Linear(d, cfg.num_policy_slots)  # learned importance

        # Recurrent gating
        self.recurrent_gate = nn.Linear(d, d, bias=False)

        self.apply(self._init_weights)
        with torch.no_grad():
            self.halt_head.bias.fill_(-8.0)  # strong early continue bias

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def initial_carry(self, batch_size: int, device: torch.device) -> Carry:
        d = self.cfg.hidden_size
        return Carry(
            z=torch.zeros(batch_size, d, device=device, dtype=self.dtype),
            policies=torch.zeros(self.cfg.num_policy_slots, batch_size, d, device=device, dtype=self.dtype),
            step=torch.zeros(batch_size, dtype=torch.long, device=device),
            halted=torch.zeros(batch_size, dtype=torch.bool, device=device),
            policy_mask=torch.zeros(self.cfg.num_policy_slots, dtype=torch.bool, device=device),
        )

    def forward(self, carry: Carry, input_ids: torch.Tensor) -> Tuple[Carry, Dict]:
        B, T = input_ids.shape
        d = self.cfg.hidden_size
        device = input_ids.device

        # === Embedding + mean pooling injection ===
        x = self.embed(input_ids) * math.sqrt(d)
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = x + self.rotary(pos)
        inject = x.mean(dim=1)  # [B, D]

        h = carry.z

        # === Inject discovered policies (with learned gates) ===
        if carry.policy_mask.any():
            active_policies = carry.policies[carry.policy_mask]  # [n_active, B, D]
            gate_logits = self.policy_gate(h)                     # [B, num_slots]
            gates = torch.sigmoid(gate_logits)[..., carry.policy_mask]
            weighted = (active_policies * gates.T.unsqueeze(-1)).sum(0)
            h = h + weighted * 0.5

        # === Tier 1 – Fast intuition ===
        z1 = h[:, :self.cfg.t1_dim]
        z1 = self.tier1(z1, inject * 0.5, steps=self.cfg.t1_steps)

        # === Tier 2 – Structured reasoning ===
        z2 = self.up1(z1)
        z2 = self.tier2(z2, inject, steps=self.cfg.t2_steps)

        # === Tier 3 – Deep precision ===
        z3 = self.up2(z2)
        new_h_raw = self.tier3(z3, inject, steps=self.cfg.t3_steps)
        new_h_raw = self.down3(new_h_raw)

        # === Verification & value ===
        verify_in = new_h_raw
        slow = (carry.step[0] % self.cfg.slow_verify_every == 0)
        if slow and self.training:
            for block in self.verifier:
                verify_in = block(verify_in, inject * 0.1)
            value = self.slow_critic(verify_in.mean(0, keepdim=True))
        else:
            value = self.fast_critic(verify_in.mean(0, keepdim=True))

        gate = torch.sigmoid(self.recurrent_gate(h))
        new_h = rms_norm(h + gate * verify_in, self.cfg.rms_eps)

        logits = self.lm_head(new_h)[:, :T]
        q = self.halt_head(new_h.mean(1))
        q_halt, q_cont = q[:, 0], q[:, 1]
        q_evolve = self.evolve_head(new_h.mean(1, keepdim=True))

        # === Latent Self-Evolution ===
        new_policy = None
        evolve_now = (q_evolve.sigmoid() > self.cfg.evolution_threshold) & self.training
        if evolve_now.any():
            noise = torch.randn(self.cfg.evolution_candidates, B, d, device=device, dtype=self.dtype) * self.cfg.evolution_noise
            candidates = new_h.unsqueeze(0) + noise
            candidates = rms_norm(candidates, self.cfg.rms_eps)
            scores = self.slow_critic(candidates.mean(-1))  # [C, B, 1]
            best = scores.squeeze(-1).argmax(0)            # [B]
            new_policy = candidates[best, torch.arange(B)]

            mask = ~carry.policy_mask
            if mask.any():
                slot = mask.nonzero(as_tuple=True)[0][0].item()
                carry.policies[slot] = new_policy.detach()
                carry.policy_mask[slot] = True

        should_halt = (carry.step >= self.cfg.max_ponder_steps) | (q_halt > q_cont)
        if self.training:
            should_halt = should_halt | (torch.rand_like(q_cont) < 0.04)

        new_carry = Carry(
            z=new_h.detach(),
            policies=carry.policies.detach(),
            step=carry.step + 1,
            halted=carry.halted | should_halt,
            policy_mask=carry.policy_mask.clone(),
        )

        outputs = {
            "logits": logits,
            "value": value.squeeze(-1),
            "new_policy": new_policy,
            "active_policies": int(carry.policy_mask.sum()),
            "step": carry.step,
        }
        if self.training:
            outputs["loss_aux"] = self.cfg.ponder_cost_weight * carry.step.float().mean()

        return new_carry, outputs
