from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rainbow import _compact, _batch_obs, to_tensor


class PointerPolicy(nn.Module):
    def __init__(self, state_dim: int, node_dim: int, hidden: int = 96,
                 ent_coef: float = 0.01):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU())
        self.node = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.ent_coef = ent_coef

    def forward(self, s, node, mask):
        h = self.h(s)
        g = self.node(node)
        logits = torch.einsum("bh,bjh->bj", h, g) / np.sqrt(h.shape[-1])
        logits = logits.masked_fill(~mask, -1e9)
        v = self.value(h).squeeze(-1)
        return logits, v


@dataclass
class Rollout:
    obs: Dict[str, np.ndarray]
    action: int
    logp: float
    value: float
    reward: float
    done: bool
    mask: np.ndarray


class PGAgent:
    def __init__(self, encoder: nn.Module, node_dim: int, kind: str = "a2c",
                 lr: float = 6e-3, gamma: float = 0.90, lam: float = 0.95,
                 rollout: int = 64, epochs: int = 4, clip: float = 0.2,
                 device: str = "cpu", reward_scale: float = 1.0,
                 entropy: float = 0.01, auto_scale: bool = True):
        assert kind in ("a2c", "ppo", "reinforce")
        self.kind = kind
        self.encoder = encoder.to(device)
        self.policy = PointerPolicy(getattr(encoder, "state_dim", 128),
                                    node_dim, ent_coef=entropy).to(device)
        self.opt = torch.optim.Adam(list(self.encoder.parameters()) +
                                    list(self.policy.parameters()), lr=lr)
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.rollout_len = rollout
        self.epochs = epochs
        self.clip = clip
        self.reward_scale = max(reward_scale, 1e-6)

        self.auto_scale = auto_scale
        self._r_ema = 0.0
        self.entropy = entropy
        self.buf: List[Rollout] = []
        self.last_value = 0.0

    @torch.no_grad()
    def _forward(self, obs, mask):
        s, node = self.encoder(to_tensor(obs, self.device))
        m = torch.as_tensor(mask[None, :], device=self.device)
        logits, v = self.policy(s, node, m)
        return logits, v

    def act(self, obs, mask, greedy: bool = False) -> int:
        if not mask.any():
            return 0
        logits, v = self._forward(obs, mask)
        logp = F.log_softmax(logits, dim=-1)[0]
        if greedy:
            a = int(torch.argmax(logp).item())
        else:
            a = int(torch.multinomial(logp.exp(), 1).item())
        self._pending = (float(logp[a].item()), float(v.item()))
        return a

    def observe(self, obs, action, reward, next_obs, done, mask) -> None:
        logp, v = getattr(self, "_pending", (0.0, 0.0))
        r = float(reward)
        if self.auto_scale:
            m = abs(r)
            if self._r_ema > 0.0:
                m = min(m, 10.0 * self._r_ema)
            self._r_ema = 0.98 * self._r_ema + 0.02 * m
            self.reward_scale = max(self._r_ema, 1e-6)
        self.buf.append(Rollout(_compact(obs), int(action), logp, v,
                                r, bool(done), mask))

    def update(self, force: bool = False) -> Optional[float]:
        if len(self.buf) < self.rollout_len and not force:
            return None
        if not self.buf:
            return None

        if force and len(self.buf) < 4:
            self.buf = []
            return None
        loss = self._update_from(self.buf)
        self.buf = []
        return loss

    def _update_from(self, buf: List[Rollout]) -> float:
        obs = _batch_obs([type("T", (), {"obs": r.obs, "next_obs": r.obs}) for r in buf], False)
        masks = torch.as_tensor(np.stack([r.mask for r in buf]).astype(bool), device=self.device)
        acts = torch.as_tensor([r.action for r in buf], device=self.device)
        rewards = torch.as_tensor([r.reward / self.reward_scale for r in buf],
                                  dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(r.done) for r in buf], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor([r.logp for r in buf], dtype=torch.float32, device=self.device)
        old_v = torch.as_tensor([r.value for r in buf], dtype=torch.float32, device=self.device)

        s, node = self.encoder(obs)
        logits, v = self.policy(s, node, masks)
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, acts[:, None]).squeeze(1)
        ent = -(logp_all.exp() * logp_all).sum(-1).mean()

        if self.kind == "ppo":
            adv = torch.zeros_like(rewards)
            gae = 0.0
            next_v = 0.0
            for t in reversed(range(len(buf))):
                nonterm = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_v * nonterm - old_v[t]
                gae = delta + self.gamma * self.lam * nonterm * gae
                adv[t] = gae
                next_v = old_v[t]
        else:
            ret = torch.zeros_like(rewards)
            run = 0.0
            for t in reversed(range(len(buf))):
                run = rewards[t] + self.gamma * run * (1.0 - dones[t])
                ret[t] = run
            adv = ret - old_v
        ret_target = adv + old_v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        if self.kind == "ppo":
            total = 0.0
            for _ in range(self.epochs):
                loss = 0.0
                ratio = (logp - old_logp).exp()
                s1 = ratio * adv
                s2 = ratio.clamp(1 - self.clip, 1 + self.clip) * adv
                pg = -torch.min(s1, s2).mean()
                vf = F.mse_loss(v, ret_target)
                loss = pg + 0.5 * vf - self.entropy * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                                         list(self.policy.parameters()), 5.0)
                self.opt.step()
                total += float(loss.item())

                s, node = self.encoder(obs)
                logits, v = self.policy(s, node, masks)
                logp = F.log_softmax(logits, dim=-1).gather(1, acts[:, None]).squeeze(1)
                ent = -(F.log_softmax(logits, -1).exp() * F.log_softmax(logits, -1)).sum(-1).mean()
            return total / self.epochs

        pg = -(logp * adv.detach()).mean()
        vf = F.mse_loss(v, ret_target.detach())
        loss = pg + 0.5 * vf - self.entropy * ent
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                                 list(self.policy.parameters()), 5.0)
        self.opt.step()
        return float(loss.item())
