from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from knowledge.transh import TransH
from .mkfr_clnet import MKFRCLNet, STATE_DIM, VIS_DIM, KG_DIM

from cloudsce.node import NODE_FEATURE_DIM as NODE_DIM
INST_DIM = 39


def _ensure_batch(obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    img = obs["img"]
    if img.dim() == 3:
        return {k: v.unsqueeze(0) for k, v in obs.items()}
    return obs


class MKFREncoder(nn.Module):
    def __init__(self, n_symbols: int, state_dim: int = STATE_DIM,
                 depth: int = 4, vis_dim: int = 96):
        super().__init__()
        self.net = MKFRCLNet(n_symbols, depth=depth, vis_dim=vis_dim)
        self.post = nn.Sequential(nn.Linear(state_dim + INST_DIM, state_dim), nn.ReLU())

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = _ensure_batch(obs)
        enc = self.net.encode(obs["img"], obs["tokens"].long())
        s = self.post(torch.cat([enc["F_M"], obs["inst"]], dim=-1))
        return s, obs["node"]

    def parts(self, img, tokens):
        return self.net.encode(img, tokens.long())


class HeatmapEncoder(nn.Module):
    def __init__(self, state_dim: int = 128, depth: int = 2):
        super().__init__()
        self.net = MKFRCLNet(16, depth=depth, vis_dim=96)
        self.proj = nn.Sequential(nn.Linear(VIS_DIM + INST_DIM, state_dim), nn.ReLU())
        self.state_dim = state_dim

    def forward(self, obs):
        obs = _ensure_batch(obs)
        fv, _ = self.net.visual(obs["img"])
        s = self.proj(torch.cat([fv, obs["inst"]], dim=-1))
        return s, obs["node"]


class VectorEncoder(nn.Module):
    def __init__(self, state_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INST_DIM + 12 + NODE_DIM, 256), nn.ReLU(),
                                 nn.Linear(256, state_dim), nn.ReLU())
        self.state_dim = state_dim

    def forward(self, obs):
        obs = _ensure_batch(obs)
        node = obs["node"]

        rem = node[..., :6]
        glob = torch.cat([rem.mean(1), rem.max(1).values], dim=-1)
        pooled = node.mean(1)
        s = self.net(torch.cat([obs["inst"], glob, pooled], dim=-1))
        return s, node


class RelationalEncoder(nn.Module):
    def __init__(self, state_dim: int = 128, dim: int = 96, heads: int = 2):
        super().__init__()
        self.embed = nn.Linear(NODE_DIM, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim + INST_DIM, state_dim), nn.ReLU())
        self.state_dim = state_dim

    def forward(self, obs):
        obs = _ensure_batch(obs)
        node = obs["node"]
        x = self.embed(node)
        kpmask = torch.zeros(node.size(0), node.size(1), dtype=torch.bool,
                             device=node.device)
        a, _ = self.attn(x, x, x, key_padding_mask=kpmask, need_weights=False)
        x = self.norm(x + a)
        s = self.proj(torch.cat([x.mean(1), obs["inst"]], dim=-1))
        return s, node


class GraphLayer(nn.Module):
    def __init__(self, dim: int, hidden: int = 96):
        super().__init__()
        self.lin = nn.Linear(dim * 2, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        msg = torch.bmm(adj, x)
        return F.relu(self.out(F.relu(self.lin(torch.cat([x, msg], dim=-1)))))


class DAGEncoder(nn.Module):
    def __init__(self, state_dim: int = 128, dim: int = 64, n_symbols: int = 512):
        super().__init__()
        self.ent = nn.Embedding(n_symbols, dim)
        self.task_emb = nn.Embedding(8, dim)
        self.layer = GraphLayer(dim, hidden=dim)
        self.node_proj = nn.Linear(NODE_DIM, dim)
        self.proj = nn.Sequential(nn.Linear(dim * 2 + INST_DIM, state_dim), nn.ReLU())
        self.state_dim = state_dim

    def forward(self, obs):
        obs = _ensure_batch(obs)
        tok = obs["tokens"].long()
        task_tokens = tok[:, 2, :12]
        x = self.ent(task_tokens.clamp(min=0))
        n = x.size(1)

        B = x.size(0)
        adj = torch.eye(n, device=x.device).unsqueeze(0).expand(B, n, n).clone()
        adj = adj + torch.triu(torch.ones(n, n, device=x.device), diagonal=1).unsqueeze(0)
        adj = adj / adj.sum(-1, keepdim=True)
        for _ in range(2):
            x = self.layer(x, adj)
        g = x.mean(1)
        nodes = self.node_proj(obs["node"]).mean(1)
        s = self.proj(torch.cat([g, nodes, obs["inst"]], dim=-1))
        return s, obs["node"]


class BiGCNEncoder(nn.Module):
    def __init__(self, state_dim: int = 128, dim: int = 64, n_symbols: int = 512):
        super().__init__()
        self.ent = nn.Embedding(n_symbols, dim)
        self.fwd = GraphLayer(dim)
        self.bwd = GraphLayer(dim)
        self.node_proj = nn.Linear(NODE_DIM, dim)
        self.proj = nn.Sequential(nn.Linear(dim * 3 + INST_DIM, state_dim), nn.ReLU())
        self.state_dim = state_dim

    def forward(self, obs):
        obs = _ensure_batch(obs)
        tok = obs["tokens"].long()
        x = self.ent(tok[:, 2, :12].clamp(min=0))
        n = x.size(1)
        B = x.size(0)
        base = torch.eye(n, device=x.device).unsqueeze(0).expand(B, n, n)
        up = torch.triu(torch.ones(n, n, device=x.device), 1).unsqueeze(0).expand(B, n, n) + base
        lo = torch.tril(torch.ones(n, n, device=x.device), -1).unsqueeze(0).expand(B, n, n) + base
        up = up / up.sum(-1, keepdim=True)
        lo = lo / lo.sum(-1, keepdim=True)
        hf = self.fwd(x, up)
        hb = self.bwd(x, lo)
        g = torch.cat([hf.mean(1), hb.mean(1)], dim=-1)
        nodes = self.node_proj(obs["node"]).mean(1)
        s = self.proj(torch.cat([g, nodes, obs["inst"]], dim=-1))
        return s, obs["node"]


class KnowledgeEncoder(nn.Module):
    def __init__(self, base: nn.Module, n_symbols: int = 512, kg_dim: int = 32,
                 state_dim: int = 128):
        super().__init__()
        self.base = base
        self.kg = TransH(n_symbols, dim=kg_dim)
        out_dim = int(getattr(base, "state_dim", state_dim))
        self.proj = nn.Sequential(nn.Linear(out_dim + kg_dim, out_dim), nn.ReLU())
        self.state_dim = out_dim

    def forward(self, obs):
        obs = _ensure_batch(obs)
        s, node = self.base(obs)
        k = self.kg(obs["tokens"].long())
        k = k.mean(dim=1)
        if k.dim() != s.dim():
            k = k.unsqueeze(0).expand(s.size(0), -1)
        return self.proj(torch.cat([s, k], dim=-1)), node
