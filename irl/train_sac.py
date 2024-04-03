import argparse
import numpy as np
import torch
import gymnasium as gym
import tianshou as ts
import os
import datetime

from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.utils.tensorboard import SummaryWriter

import envs # To register environments
import utils


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--resume-path", type=str, default=None)

    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--collect-only", default=False, action="store_true")

    return parser.parse_args()

def main(args: argparse.Namespace = get_args()):
    # System

    device = args.device

    utils.manual_seed_all(args.seed)

    # Environment

    train_env = gym.make(args.env)
    test_env = gym.make(args.env)

    # Model

    state_shape = train_env.observation_space.shape or train_env.observation_space.n
    action_shape = test_env.action_space.shape or test_env.action_space.n
    min_action = np.min(train_env.action_space.low)
    max_action = np.max(train_env.action_space.high)

    print(f"State shape: {state_shape}, Action shape: {action_shape}")
    print(f"Action range: {min_action} ~ {max_action}")

    actor = Net(state_shape, hidden_sizes=args.hidden_sizes, device=device)
    actor = ActorProb(actor, action_shape, max_action=max_action, unbounded=True, conditioned_sigma=True, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    critic1 = Net(state_shape, action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=device)
    critic1 = Critic(critic1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    critic2 = Net(state_shape, action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=device)
    critic2 = Critic(critic2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # Logging

    env_name = args.env.split(os.path.sep)[-1]
    log_name = f"{args.env}-{datetime.datetime.now():%y%m%d-%H%M%S}"
    log_path = os.path.join(args.logdir, log_name)

    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))

    if args.logger == "wandb":
        logger = ts.utils.WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
        logger.load(writer)
    elif args.logger == "tensorboard":
        logger = ts.utils.TensorboardLogger(writer=writer)

    logger = ts.utils.TensorboardLogger(writer=SummaryWriter(), train_interval=100)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, f"{env_name}-policy.pth"))

    # Policy
    
    policy = ts.policy.SACPolicy(
        actor=actor, 
        actor_optim=actor_optim, 
        critic=critic1, 
        critic_optim=critic1_optim, 
        critic2=critic2, 
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=train_env.action_space,
        )
    
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=device))
    
    # Train
    
    buffer = ts.data.ReplayBuffer(size=args.buffer_size)

    train_collector = ts.data.Collector(policy, train_env, buffer, exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_env)
    train_collector.collect(n_step=args.start_timesteps, random=True, reset_before_collect=True) # TODO What is this?

    if not args.collect_only:
        result = ts.trainer.OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=1,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        ).run()
        result.pprint_asdict()



    # Evaluate

    policy.eval()

    eval_buffer = ts.data.ReplayBuffer(size=args.buffer_size)
    eval_collector = ts.data.Collector(policy, test_env, buffer=eval_buffer, exploration_noise=True)
    
    result = eval_collector.collect(n_episode=10, render=args.render, reset_before_collect=True)
    
    eval_buffer.save_hdf5(os.path.join(log_path, f"{env_name}-traj.hdf5"))

if __name__ == "__main__":
    main()
