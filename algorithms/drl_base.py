from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from cloudsce.env import CloudsceEnv, Observation
from models.pg import PGAgent
from models.rainbow import RainbowAgent
from .base import Scheduler


class DRLScheduler(Scheduler):
    trainable = True
    family = "drl"

    def __init__(self, encoder, node_dim: int, kind: str = "rainbow",
                 state_dim: int = 128, seed: int = 0, lr: float = 6e-3,
                 gamma: float = 0.90, eps_start: float = 1.0,
                 eps_end: float = 0.05, eps_decay: float = 0.995,
                 update_every: int = 12, rollout: int = 64,
                 epochs: int = 4, batch: int = 8, buffer_size: int = 12000,
                 reward_scale: float = 1.0, target_sync: int = 200,
                 use_prior: bool = False, prior_init: float = 0.0,
                 adv_reg: float = 0.0, n_step: int = 1, **kw):
        super().__init__(seed, **kw)
        self.kind = kind
        self.reward_scale = reward_scale
        self.update_every = update_every
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self._steps = 0
        self._last_obs: Optional[Dict[str, np.ndarray]] = None
        self._eval_mode = False
        self.encoder = encoder
        self.node_dim = node_dim
        self.state_dim = state_dim

        self.use_prior = use_prior
        self._agent_kw = dict(gamma=gamma, lr=lr, batch=batch,
                              buffer_size=buffer_size, rollout=rollout,
                              epochs=epochs, reward_scale=reward_scale,
                              target_sync=target_sync, use_prior=use_prior,
                              prior_init=prior_init, adv_reg=adv_reg,
                              n_step=n_step)
        self.agent = self._make_agent()
        self.losses = []

    def _make_agent(self):
        kw = self._agent_kw
        if self.kind == "rainbow":
            return RainbowAgent(self.encoder, self.node_dim, self.state_dim,
                                gamma=kw["gamma"], lr=kw["lr"], batch=kw["batch"],
                                buffer_size=kw["buffer_size"],
                                reward_scale=kw["reward_scale"],
                                target_sync=kw["target_sync"],
                                use_prior=kw.get("use_prior", False),
                                prior_init=kw.get("prior_init", 0.0),
                                adv_reg=kw.get("adv_reg", 0.0),
                                n_step=int(kw.get("n_step", 1)),
                                device="cpu")
        return PGAgent(self.encoder, self.node_dim, kind=self.kind, lr=kw["lr"],
                       gamma=kw["gamma"], rollout=kw["rollout"], epochs=kw["epochs"],
                       reward_scale=kw["reward_scale"], device="cpu")

    def reset(self, env: CloudsceEnv) -> None:
        super().reset(env)
        self._last_obs = None
        self._steps = 0

    def reward_transform(self, reward: float, info: Dict) -> float:
        return reward

    def preprocess(self, state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return state

    def act(self, obs: Observation) -> int:
        if not obs.mask.any():
            return 0
        state = self.preprocess(obs.as_state())
        if self.kind == "rainbow":
            eps = 0.0 if self._eval_mode else self.eps
            a = self.agent.act(state, obs.mask, epsilon=eps)
        else:
            a = self.agent.act(state, obs.mask, greedy=self._eval_mode)
        self._last_obs = state
        return int(a)

    def observe(self, obs, action, reward, next_obs, done, info) -> None:
        prev = self._last_obs
        if prev is None:
            return
        r = self.reward_transform(float(reward), info or {})
        self.agent.observe(prev, action, r,
                           self.preprocess(next_obs.as_state()), done, next_obs.mask)
        self._last_obs = None
        self._steps += 1
        if self.kind == "rainbow":
            self.eps = max(self.eps_end, self.eps * self.eps_decay)
            if self._steps % self.update_every == 0:
                loss = self.agent.update()
                if loss is not None:
                    self.losses.append(loss)
        else:
            if len(self.agent.buf) >= self.agent.rollout_len:
                loss = self.agent.update()
                if loss is not None:
                    self.losses.append(loss)

    def end_episode(self) -> None:
        if self.kind != "rainbow":
            loss = self.agent.update(force=True)
            if loss is not None:
                self.losses.append(loss)
