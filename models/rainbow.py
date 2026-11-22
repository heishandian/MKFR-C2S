from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RainbowHead(nn.Module):
    def __init__(self, state_dim: int, node_dim: int, hidden: int = 96,
                 n_atoms: int = 51, vmin: float = -2.0, vmax: float = 6.0,
                 use_prior: bool = False, prior_init: float = 0.0):
        super().__init__()
        self.n_atoms = n_atoms
        self.use_prior = use_prior
        self.register_buffer("support", torch.linspace(vmin, vmax, n_atoms))
        self.h = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU())
        self.node = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, n_atoms)

        if use_prior:
            self.w_prior = nn.Parameter(torch.tensor(float(prior_init)))

    def forward(self, s: torch.Tensor, node: torch.Tensor,
                prior: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.h(s)
        g = self.node(node)
        logits = self.value(h)
        p = torch.softmax(logits, dim=-1)
        v = (p * self.support).sum(-1, keepdim=True)
        adv = torch.einsum("bh,bjh->bj", h, g) / np.sqrt(h.shape[-1])
        if self.use_prior and prior is not None:
            if prior.dim() == 1:
                prior = prior.unsqueeze(0)
            adv = adv + self.w_prior * prior.to(adv.dtype)
        adv = adv - adv.mean(dim=-1, keepdim=True)
        return v + adv, logits


@dataclass
class Transition:
    obs: Dict[str, np.ndarray]
    action: int
    reward: float
    next_obs: Dict[str, np.ndarray]
    done: bool
    mask: np.ndarray

    n: int = 1


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buf: List[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, obs, action, reward, next_obs, done, mask, n: int = 1) -> None:
        tr = Transition(_compact(obs), int(action), float(reward),
                        _compact(next_obs), bool(done), mask, int(n))
        if len(self.buf) < self.capacity:
            self.buf.append(tr)
        else:
            self.buf[self.pos] = tr
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch: int, rng: random.Random):
        idx = rng.sample(range(len(self.buf)), min(batch, len(self.buf)))
        return [self.buf[i] for i in idx]


def to_tensor(obs: Dict[str, np.ndarray], device: str = "cpu") -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            t = v
        elif v.dtype == np.uint8:
            t = torch.as_tensor(v, dtype=torch.float32).div_(255.0)
        elif k == "tokens":
            t = torch.as_tensor(v, dtype=torch.long)
        else:
            t = torch.as_tensor(v.astype(np.float32))
        want = {"img": 4, "delta": 4, "node": 3, "inst": 2, "mask": 2,
                "tokens": 3}.get(k)
        if want is not None and t.dim() == want - 1:
            t = t.unsqueeze(0)
        out[k] = t.to(device)
    return out


def _compact(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in obs.items():
        if k in ("img", "delta"):
            out[k] = np.clip(v * 255.0, 0, 255).astype(np.uint8)
        elif k in ("node", "prior"):
            out[k] = v.astype(np.float16)
        else:
            out[k] = v
    return out


def collate(trans: List[Transition], key: str, nxt: bool = False) -> torch.Tensor:
    arrs = [(t.next_obs if nxt else t.obs)[key] for t in trans]
    x = np.stack(arrs)
    if x.dtype == np.uint8:
        return torch.as_tensor(x, dtype=torch.float32).div_(255.0)
    return torch.as_tensor(x.astype(np.float32))


class RainbowAgent:
    def __init__(self, encoder: nn.Module, node_dim: int, state_dim: int,
                 gamma: float = 0.90, lr: float = 6e-3, n_atoms: int = 51,
                 vmin: float = -20.0, vmax: float = 20.0, batch: int = 16,
                 buffer_size: int = 20000, device: str = "cpu",
                 reward_scale: float = 1.0, target_sync: int = 200,
                 auto_scale: bool = True, use_prior: bool = False,
                 prior_init: float = 0.0, c51_weight: float = 0.5,
                 adv_reg: float = 0.0, n_step: int = 1):
        self.encoder = encoder.to(device)
        self.use_prior = use_prior
        self.head = RainbowHead(state_dim, node_dim, n_atoms=n_atoms,
                                vmin=vmin, vmax=vmax, use_prior=use_prior,
                                prior_init=prior_init).to(device)
        self.target_enc = _clone(encoder)
        self.target_head = _clone(self.head)
        self.params = list(self.encoder.parameters()) + list(self.head.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.gamma = gamma
        self.n_atoms = n_atoms
        self.device = device
        self.batch = batch
        self.buf = ReplayBuffer(buffer_size)
        self.rng = random.Random(0)
        self.reward_scale = max(reward_scale, 1e-6)
        self.auto_scale = auto_scale
        self._r_ema = 0.0
        self.steps = 0
        self.target_sync = target_sync

        self.c51_weight = float(c51_weight)

        self.adv_reg = float(adv_reg)

        self.n_step = max(1, int(n_step))
        self._pend: List[Transition] = []

    def _state(self, obs: Dict[str, np.ndarray]):
        return self.encoder(to_tensor(obs, self.device))

    def _prior(self, obs: Dict[str, np.ndarray]) -> Optional[torch.Tensor]:
        if not self.use_prior or "prior" not in obs:
            return None
        p = obs["prior"]
        if not isinstance(p, torch.Tensor):
            p = torch.as_tensor(np.asarray(p, dtype=np.float32))
        return p.to(self.device)

    @torch.no_grad()
    def q_values(self, obs: Dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        s, node = self._state(obs)
        q, _ = self.head(s, node, self._prior(obs))
        q = q.cpu().numpy()[0]
        self.encoder.train()
        q = q.copy()
        q[~mask] = -1e9
        return q

    def act(self, obs, mask, epsilon: float = 0.05) -> int:
        if np.random.rand() < epsilon:
            cand = np.where(mask)[0]
            return int(cand[np.random.randint(len(cand))]) if cand.size else 0
        q = self.q_values(obs, mask)
        return int(np.argmax(q))

    def observe(self, obs, action, reward, next_obs, done, mask) -> None:
        r = float(reward)
        if self.n_step == 1:
            if self.auto_scale:
                self._track_scale(abs(r))
            self.buf.add(obs, action, r, next_obs, done, mask)
            return
        self._pend.append((obs, action, r, next_obs, done, mask))
        if done:
            while self._pend:
                self._emit_n_step()
        elif len(self._pend) >= self.n_step:
            self._emit_n_step()

    def _emit_n_step(self) -> None:
        if not self._pend:
            return
        s0, a0 = self._pend[0][0], self._pend[0][1]
        R, disc, k = 0.0, 1.0, 0
        obs_last, done_last, mask_last = self._pend[0][3], self._pend[0][4], self._pend[0][5]
        for k, (_, _, rk, nxt, dk, mk) in enumerate(self._pend[:self.n_step]):
            R += disc * rk
            disc *= self.gamma
            obs_last, done_last, mask_last = nxt, dk, mk
            if dk:
                break

        if self.auto_scale:
            self._track_scale(abs(R))
        self.buf.add(s0, a0, R, obs_last, done_last, mask_last, n=k + 1)
        self._pend.pop(0)

    def _track_scale(self, mag: float) -> None:
        if self._r_ema > 0.0:
            mag = min(mag, 10.0 * self._r_ema)
        self._r_ema = 0.98 * self._r_ema + 0.02 * mag
        self.reward_scale = max(self._r_ema, 1e-3)

    def update(self) -> Optional[float]:
        if len(self.buf) < max(64, self.batch):
            return None
        trans = self.buf.sample(self.batch, self.rng)
        obs = _batch_obs(trans, False)
        nobs = _batch_obs(trans, True)
        a = torch.as_tensor([t.action for t in trans], dtype=torch.long, device=self.device)
        r = torch.as_tensor([t.reward for t in trans], dtype=torch.float32,
                            device=self.device) / self.reward_scale
        d = torch.as_tensor([float(t.done) for t in trans], dtype=torch.float32, device=self.device)
        nmask = torch.as_tensor(np.stack([t.next_obs["mask"] for t in trans]), device=self.device)

        s, node = self.encoder(obs)
        q, logits = self.head(s, node, self._prior(obs))
        q_a = q.gather(1, a[:, None]).squeeze(1)
        logp_a = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            ns, nnode = self.target_enc(nobs)
            nprior = self._prior(nobs)
            nq, nlogits = self.target_head(ns, nnode, nprior)
            nq = nq.masked_fill(~nmask, -1e9)

            oq_next, _ = self.head(*self.encoder(nobs), nprior)
            oq_next = oq_next.masked_fill(~nmask, -1e9)
            best = oq_next.argmax(dim=-1)
            next_q = nq.gather(1, best[:, None]).squeeze(1)

            nsteps = torch.as_tensor([float(t.n) for t in trans],
                                     dtype=torch.float32, device=self.device)
            disc = self.gamma ** nsteps
            y = r + disc * (1.0 - d) * next_q

            nv = (torch.softmax(nlogits, dim=-1) * self.head.support).sum(-1)
            y_v = r + disc * (1.0 - d) * nv

        q_loss = F.smooth_l1_loss(q_a, y)

        support = self.head.support
        step = support[1] - support[0]
        y_v = y_v.clamp(support[0], support[-1])
        b = (y_v - support[0]) / step
        lo = b.floor().long().clamp(0, self.n_atoms - 1)
        hi = b.ceil().long().clamp(0, self.n_atoms - 1)
        frac = (b - b.floor()).clamp(0, 1)
        target_dist = torch.zeros(len(trans), self.n_atoms, device=self.device)
        target_dist.scatter_add_(1, lo[:, None], (1 - frac)[:, None])
        target_dist.scatter_add_(1, hi[:, None], frac[:, None])
        loss_c51 = -(target_dist * logp_a).sum(-1).mean()

        loss = q_loss + self.c51_weight * loss_c51
        if self.adv_reg and self.use_prior:
            bmask = torch.as_tensor(np.stack([t.obs["mask"] for t in trans]),
                                    device=self.device)
            adv = (q - (torch.softmax(logits, dim=-1) * support).sum(-1, keepdim=True))
            loss = loss + self.adv_reg * ((adv * bmask).pow(2).sum()
                                          / bmask.sum().clamp(min=1))

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, 5.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self._sync()
        return float(loss.item())

    def _sync(self) -> None:
        self.target_enc.load_state_dict(self.encoder.state_dict())
        self.target_head.load_state_dict(self.head.state_dict())


def _clone(m: nn.Module) -> nn.Module:
    import copy
    c = copy.deepcopy(m)
    for p in c.parameters():
        p.requires_grad_(False)
    return c


def _batch_obs(trans: List[Transition], nxt: bool) -> Dict[str, torch.Tensor]:
    out = {}
    keys = trans[0].obs.keys()
    for k in keys:
        arrs = [(t.next_obs if nxt else t.obs)[k] for t in trans]
        x = np.stack(arrs)
        if x.dtype == np.uint8:
            out[k] = torch.as_tensor(x, dtype=torch.float32).div_(255.0)
        else:
            out[k] = torch.as_tensor(x.astype(np.float32))
    return out
