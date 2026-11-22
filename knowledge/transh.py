from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctssm import Triple


class TransH(nn.Module):
    def __init__(self, n_symbols: int, dim: int = 32, margin: float = 1.0,
                 n_neg: int = 10):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.n_neg = n_neg
        self.ent = nn.Embedding(n_symbols, dim)
        self.rel = nn.Embedding(n_symbols, dim)
        self.norm = nn.Embedding(n_symbols, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        nn.init.xavier_uniform_(self.norm.weight)

    def project(self, e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w = F.normalize(w, p=2, dim=-1)
        return e - (e * w).sum(-1, keepdim=True) * w

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eh, er, et = self.ent(h), self.rel(r), self.ent(t)
        wr = self.norm(r)
        hp = self.project(eh, wr)
        tp = self.project(et, wr)
        return -torch.norm(hp + er - tp, p=2, dim=-1)

    def triple_repr(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eh, er, et = self.ent(h), self.rel(r), self.ent(t)
        wr = self.norm(r)
        return (self.project(eh, wr) + er + self.project(et, wr)) / 3.0

    def forward(self, tokens: torch.Tensor, view_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        B, V, L = tokens.shape
        n_trip = L // 3
        tri = tokens[:, :, : n_trip * 3].reshape(B, V, n_trip, 3)
        h, r, t = tri[..., 0], tri[..., 1], tri[..., 2]
        emb = self.triple_repr(h.reshape(-1), r.reshape(-1), t.reshape(-1))
        emb = emb.reshape(B, V, n_trip, -1)
        valid = (h > 0).float().unsqueeze(-1)
        cnt = valid.sum(2).clamp(min=1.0)
        return (emb * valid).sum(2) / cnt

    def margin_loss(self, pos: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    n_symbols: int) -> torch.Tensor:
        h, r, t = pos
        sh = self.score(h, r, t)
        neg_t = torch.randint(2, n_symbols, t.shape, device=t.device)
        st = self.score(h, r, neg_t)
        neg_h = torch.randint(2, n_symbols, h.shape, device=h.device)
        sh2 = self.score(neg_h, r, t)
        return F.relu(self.margin - sh + st).mean() + F.relu(self.margin - sh + sh2).mean()


class KGEncoder:
    def __init__(self, vocab, dim: int = 32, device: str = "cpu", seed: int = 0):
        self.vocab = vocab
        self.device = device
        torch.manual_seed(seed)
        self.model = TransH(len(vocab), dim=dim).to(device)
        self.trained = False

    def parameters(self):
        return self.model.parameters()

    def encode(self, token_ids: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(token_ids), dtype=torch.long, device=self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return self.model(x)

    def pretrain(self, corpus: Sequence[Triple], epochs: int = 30, lr: float = 1e-2,
                 batch: int = 512) -> List[float]:
        if not corpus:
            return []
        ids = np.asarray([[self.vocab.id(t.head), self.vocab.id(t.relation),
                           self.vocab.id(t.tail)] for t in corpus], dtype=np.int64)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses: List[float] = []
        for ep in range(epochs):
            perm = np.random.permutation(len(ids))
            tot = 0.0
            for i in range(0, len(ids), batch):
                sel = perm[i:i + batch]
                b = torch.as_tensor(ids[sel], dtype=torch.long, device=self.device)
                loss = self.model.margin_loss((b[:, 0], b[:, 1], b[:, 2]), len(self.vocab))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.item()) * len(sel)
            losses.append(tot / len(ids))
        self.trained = True
        return losses
