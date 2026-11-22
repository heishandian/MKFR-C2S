from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from knowledge.transh import TransH

IMG = 64
OBS_CH = 12
STATE_DIM = 256
VIS_DIM = 128
KG_DIM = 32


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, dim: int = 96, patch: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class VisualEncoder(nn.Module):
    def __init__(self, in_ch: int = OBS_CH, dim: int = 96, depth: int = 4,
                 heads: int = 4, pool: int = 2, patch: int = 8):
        super().__init__()
        self.pool = nn.AvgPool2d(pool) if pool > 1 else nn.Identity()
        self.adapter = nn.Conv2d(in_ch, 3, kernel_size=1)
        self.patch = PatchEmbed(3, dim, patch)
        n_tokens = (IMG // pool // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        self.proj = nn.Sequential(nn.Linear(dim, VIS_DIM), nn.ReLU(),
                                  nn.LayerNorm(VIS_DIM))

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.adapter(self.pool(img))
        x = self.patch(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1) + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        cls, patches = x[:, 0], x[:, 1:]
        return self.proj(cls), patches


class CrossAttention(nn.Module):
    def __init__(self, kg_dim: int = KG_DIM, vis_dim: int = 96, heads: int = 4):
        super().__init__()
        self.q = nn.Linear(kg_dim, vis_dim)
        self.k = nn.Linear(vis_dim, vis_dim)
        self.v = nn.Linear(vis_dim, vis_dim)
        self.attn = nn.MultiheadAttention(vis_dim, heads, batch_first=True)
        self.out = nn.Linear(vis_dim, VIS_DIM)

    def forward(self, fk: torch.Tensor, patches: torch.Tensor,
                patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q(fk)
        k = self.k(patches)
        v = self.v(patches)
        a, _ = self.attn(q, k, v, need_weights=False)
        return self.out(a)


class MKFRCLNet(nn.Module):
    def __init__(self, n_symbols: int, tokens_per_view: int = 24,
                 depth: int = 4, vis_dim: int = 96, kg_dim: int = KG_DIM):
        super().__init__()
        self.visual = VisualEncoder(depth=depth, dim=vis_dim)
        self.kg = TransH(n_symbols, dim=kg_dim)
        self.xattn = CrossAttention(kg_dim, vis_dim)

        self.k_proj = nn.Linear(kg_dim, VIS_DIM)
        self.fuse = nn.Sequential(nn.Linear(VIS_DIM + 4 * kg_dim, STATE_DIM),
                                  nn.ReLU(),
                                  nn.Linear(STATE_DIM, STATE_DIM))
        self.tokens_per_view = tokens_per_view

    def encode(self, img: torch.Tensor, tokens: torch.Tensor
               ) -> Dict[str, torch.Tensor]:
        fv, patches = self.visual(img)
        fk = self.kg(tokens)
        fk_attn = self.xattn(fk, patches)

        fused = self.fuse(torch.cat([fk_attn.mean(1), fk.flatten(1)], dim=-1))
        return {"F_V": fv, "F_K": fk, "F_Kp": self.k_proj(fk),
                "F_attn": fk_attn, "F_M": fused, "patches": patches}

    def forward(self, img: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.encode(img, tokens)["F_M"]


def info_nce(anchor: torch.Tensor, positive: torch.Tensor,
             negatives: Optional[torch.Tensor] = None, tau: float = 0.07) -> torch.Tensor:
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    if negatives is None:
        logits = a @ p.t() / tau
    else:
        n = F.normalize(negatives, dim=-1)
        logits = torch.einsum("bd,bkd->bk", a, n) / tau
        return -F.log_softmax(logits, dim=-1)[:, 0].mean()
    target = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, target)


def iic_loss(fv1: torch.Tensor, fv2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    return 0.5 * (contrastive_pair(fv1, fv2, tau) + contrastive_pair(fv2, fv1, tau))


def contrastive_pair(a: torch.Tensor, b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / tau
    return F.cross_entropy(logits, torch.arange(a.size(0), device=a.device))


def ivc_loss(fv: torch.Tensor, fk: torch.Tensor, weights=(0.25, 0.25, 0.30, 0.20),
             tau: float = 0.07) -> torch.Tensor:
    total = 0.0
    for v, w in enumerate(weights):
        total = total + w * contrastive_pair(fv, fk[:, v, :], tau)
    return total


def cl_loss(fv1, fv2, fv_anchor, fk, lam_m: float = 0.45,
            weights=(0.25, 0.25, 0.30, 0.20), tau: float = 0.07) -> Dict[str, torch.Tensor]:
    l_iic = iic_loss(fv1, fv2, tau)
    l_ivc = ivc_loss(fv_anchor, fk, weights, tau)
    return {"L_IIC": l_iic, "L_IVC": l_ivc, "L_CL": lam_m * l_ivc + (1 - lam_m) * l_iic}
