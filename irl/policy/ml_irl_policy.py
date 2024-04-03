from dataclasses import dataclass
from typing import Any, Dict, Literal

import numpy as np
import torch
import gymnasium as gym

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.exploration import BaseNoise
from tianshou.policy import SACPolicy
from tianshou.policy.modelfree.sac import SACTrainingStats
from tianshou.utils.net.continuous import ActorProb
from tianshou.policy.base import TLearningRateScheduler


@dataclass(kw_only=True)
class MLIRLTrainingStats(SACTrainingStats):
    ml_irl_loss: float

class MLIRLPolicy(SACPolicy):
    """
    Implementation of Maximum-Likelihood Inverse Reinforcement Learning with Finite-Time Guarantees. arXiv:2210.01808.

    Built on top of Tianshou's SACPolicy.

    :param actor: the actor network following the rules (s -> dist_input_BD)
    :param actor_optim: the optimizer for actor network.
    :param critic: the first critic network. (s, a -> Q(s, a))
    :param critic_optim: the optimizer for the first critic network.
    :param action_space: Env's action space. Should be gym.spaces.Box.
    :param reward_model: the reward model network. (s, a -> r)
    :param reward_optim: the optimizer for the reward model network.
    :param expert_buffer: the expert replay buffer.
    :param critic2: the second critic network. (s, a -> Q(s, a)).
        If None, use the same network as critic (via deepcopy).
    :param critic2_optim: the optimizer for the second critic network.
        If None, clone critic_optim to use for critic2.parameters().
    :param tau: param for soft update of the target network.
    :param gamma: discount factor, in [0, 1].
    :param alpha: entropy regularization coefficient.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided,
        then alpha is automatically tuned.
    :param exploration_noise: add noise to action for exploration.
        This is useful when solving "hard exploration" problems.
        "default" is equivalent to GaussianNoise(sigma=0.1).
    :param deterministic_eval: whether to use deterministic action
        (mode of Gaussian policy) in evaluation mode instead of stochastic
        action sampled by the policy. Does not affect training.
    :param action_scaling: whether to map actions from range [-1, 1]
        to range[action_spaces.low, action_spaces.high].
    :param action_bound_method: method to bound action to range [-1, 1],
        can be either "clip" (for simply clipping the action)
        or empty string for no bounding. Only used if the action_space is continuous.
    :param observation_space: Env's observation space.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate
        in optimizer in each policy.update()

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module | ActorProb,
        actor_optim: torch.optim.Optimizer,
        critic: torch.nn.Module,
        critic_optim: torch.optim.Optimizer,
        action_space: gym.Space,
        reward_model: torch.nn.Module,
        reward_optim: torch.optim.Optimizer,
        expert_buffer: ReplayBuffer,
        critic2: torch.nn.Module | None = None,
        critic2_optim: torch.optim.Optimizer | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: float | tuple[float, torch.Tensor, torch.optim.Optimizer] = 0.2,
        exploration_noise: BaseNoise | Literal["default"] | None = None,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: Literal["clip"] | None = "clip",
        observation_space: gym.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic,
            critic_optim=critic_optim,
            action_space=action_space,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            estimation_step=1, # TODO Using n-step might break the reward model
            exploration_noise=exploration_noise,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=observation_space,
            lr_scheduler=lr_scheduler,
        )

        self.reward_model = reward_model
        self.reward_optim = reward_optim
        self.expert_buffer = expert_buffer

    def process_fn(self, 
                   batch: Batch, 
                   buffer: ReplayBuffer, 
                   indices: np.ndarray) -> Batch:
        """
        Preprocess the data from the replay buffer. This function is called before
        the data is used to train the policy.

        We need to compute the reward for the given batch and overwrite the reward
        in the batch.
        """
        with torch.no_grad():
            batch.reward = to_numpy(self.reward(batch))
        return super().process_fn(batch, buffer, indices)

    def reward(self, batch: Batch) -> torch.Tensor:
        """
        Compute the reward for the given batch.
        """
        obs = to_torch(batch.obs, device=self.reward_model.device)
        act = to_torch(batch.act, device=self.reward_model.device)

        return self.reward_model(torch.cat([obs, act], dim=-1))

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # Evaluate the policy under the current reward model
        # Update the policy

        sac_loss_stats = super().learn(batch, **kwargs)

        # Sample an expert trajectory

        expert_batch = self.expert_buffer.sample(len(batch))[0]

        # Compute the rewards

        agent_reward = self.reward(batch)
        expert_reward = self.reward(expert_batch)

        # Compute the loss (Gradient ascent)

        loss = agent_reward.mean() - expert_reward.mean()

        # Update the reward model

        self.reward_optim.zero_grad()
        loss.backward()
        self.reward_optim.step()

        return MLIRLTrainingStats(**sac_loss_stats.__dict__, ml_irl_loss=loss.item())
        



        





        

       
