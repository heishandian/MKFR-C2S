from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cloudsce.env import CloudsceEnv, Observation
from knowledge.kg_builder import TOKENS_PER_VIEW
from models.encoders import NODE_DIM, MKFREncoder
from models.mkfr_clnet import cl_loss
from models.rainbow import RainbowAgent
from .drl_base import DRLScheduler


class MKFRC2S(DRLScheduler):
    name = "MKFR-C2S"
    family = "drl"

    def __init__(self, seed: int = 0, n_symbols: int = 512,
                 lam_m: float = 0.45, weights: Tuple[float, ...] = (0.25, 0.25, 0.30, 0.20),
                 depth: int = 4, freeze_backbone: bool = True,
                 view_mask: Optional[Sequence[bool]] = None, **kw):
        enc = MKFREncoder(n_symbols, state_dim=256, depth=depth)

        use_prior = bool(kw.pop("use_prior", True))
        super().__init__(enc, NODE_DIM, kind="rainbow", state_dim=256,
                         seed=seed, buffer_size=kw.pop("buffer_size", 12000),
                         use_prior=use_prior, **kw)
        self.lam_m = lam_m
        self.weights = weights
        self.freeze_backbone = freeze_backbone
        self.pretrained = False
        self.pretrain_loss: List[float] = []

        self.view_mask = tuple(view_mask) if view_mask is not None else (True,) * 4

    def preprocess(self, state):
        if all(self.view_mask):
            return state
        out = dict(state)
        t = np.array(out["tokens"], copy=True)
        for v, on in enumerate(self.view_mask):
            if not on:
                t[v] = 0
        out["tokens"] = t
        return out

    def pretrain(self, pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
                 epochs: int = 8, batch: int = 32, lr: float = 2e-3,
                 verbose: bool = False) -> List[float]:
        if not pairs:
            return []
        imgs = torch.as_tensor(np.stack([p[0] for p in pairs]).astype(np.float32))
        toks = torch.as_tensor(np.stack([p[1] for p in pairs]).astype(np.int64))
        opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        n = imgs.size(0)
        hist: List[float] = []
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                if idx.numel() < 4:
                    continue
                x1 = self._augment(imgs[idx])
                x2 = self._augment(imgs[idx])
                t = toks[idx]
                p1 = self.encoder.parts(x1, t)
                p2 = self.encoder.parts(x2, t)
                loss = cl_loss(p1["F_V"], p2["F_V"], p1["F_V"], p1["F_Kp"],
                               lam_m=self.lam_m, weights=self.weights)["L_CL"]
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
                opt.step()
                tot += float(loss.item()) * idx.numel()
            hist.append(tot / max(n, 1))
            if verbose:
                print(f"    [Stage I] epoch {ep + 1}/{epochs}  L_CL={hist[-1]:.4f}")
        self.pretrained = True
        self.pretrain_loss = hist
        return hist

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        b = x.clone()
        shift = int(torch.randint(1, 4, (1,)).item())
        b = torch.roll(b, shifts=shift, dims=3)
        b = torch.roll(b, shifts=int(torch.randint(0, 3, (1,)).item()), dims=2)
        if torch.rand(1).item() < 0.5:
            ch = int(torch.randint(0, b.size(1), (1,)).item())
            b[:, ch] = b[:, ch] * 0.5
        b = (b + 0.02 * torch.randn_like(b)).clamp(0, 1)
        return b

    def _freeze(self) -> None:
        if not self.freeze_backbone:
            return
        net = self.encoder.net
        for module in (net.visual.adapter, net.visual.patch, net.visual.blocks,
                       net.visual.norm, net.kg):
            for p in module.parameters():
                p.requires_grad_(False)
        net.visual.cls.requires_grad_(False)
        net.visual.pos.requires_grad_(False)

    def reset(self, env: CloudsceEnv) -> None:
        if not getattr(self, "_frozen", False):
            self._freeze()

            self.agent = self._make_agent()
            self._frozen = True
        super().reset(env)

    def reward_transform(self, reward: float, info: Dict) -> float:
        n = self.env.cluster.n if self.env is not None else 1
        return float(reward / max(n, 1))

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
