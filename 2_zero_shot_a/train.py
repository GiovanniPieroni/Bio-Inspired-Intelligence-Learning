import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import time

import sys
import pathlib

current_dir = pathlib.Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
core_utils_dir = project_root / "core_utils"

for d in [str(current_dir), str(project_root), str(core_utils_dir)]:
    if d not in sys.path:
        sys.path.append(d)

from agent import Agent
from Models.FFN.model import Actor, Critic
from SatelliteEnv_P_only import SatelliteEnv
from tudatpy.astro import element_conversion
from config import load_config
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prioritized_replay import PrioritizedReplayBuffer

cfg = load_config()

initial_keplerian = np.array(
    [
        cfg.orbit.semi_major_axis,
        cfg.orbit.eccentricity,
        cfg.orbit.inclination_rad,
        cfg.orbit.arg_of_periapsis_rad,
        cfg.orbit.raan_rad,
        cfg.orbit.true_anomaly_rad,
    ]
)

cartesian_state = element_conversion.keplerian_to_cartesian(
    initial_keplerian,
    cfg.orbit.mu_m3_s2,
)

target_keplerian = np.array(
    [
        cfg.orbit.semi_major_axis + 100000.0,
        cfg.orbit.eccentricity,
        cfg.orbit.inclination_rad,
        cfg.orbit.arg_of_periapsis_rad,
        cfg.orbit.raan_rad,
        cfg.orbit.true_anomaly_rad,
    ]
)

target_mee = element_conversion.keplerian_to_mee(target_keplerian)
T_orbital = 2 * np.pi * np.sqrt(cfg.orbit.semi_major_axis**3 / cfg.orbit.mu_m3_s2)


@torch.no_grad()
def evaluate_policy(actor, env, num_episodes=100, max_steps=2):
    rewards = []
    successes = []
    final_p_errors = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0.0

        while not done and step < max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = actor(state_tensor).squeeze(0).numpy()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
            step += 1

        rewards.append(episode_reward)
        successes.append(bool(terminated))
        final_p_errors.append(
            float(env.equinoctial_parameters[0] - env.target_orbit[0])
        )

    rewards = np.array(rewards)
    abs_p_errors = np.abs(np.array(final_p_errors))

    print("\n--- Deterministic evaluation (noise off, learning off) ---")
    print(f"Episodes:       {num_episodes}")
    print(f"Success rate:   {100.0 * np.mean(successes):.1f}%")
    print(f"Mean reward:    {rewards.mean():.4f} (std {rewards.std():.4f})")
    print(
        f"Mean |p error|: {abs_p_errors.mean():.2f} m (max {abs_p_errors.max():.2f} m)"
    )
    print("------------------------------------------------------------\n")

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(rewards.mean()),
        "mean_abs_p_error": float(abs_p_errors.mean()),
    }


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(seed: int, base_save_dir: str = None):
    if base_save_dir is None:
        base_save_dir = str(current_dir / "results")
    set_seed(seed)

    save_dir = os.path.join(base_save_dir, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    STATE_DIM = cfg.rl.state_dim
    ACTION_DIM = cfg.rl.action_dim
    MAX_EPISODES = cfg.rl.num_episodes
    MAX_STEPS = cfg.simulation.steps

    GAMMA = cfg.rl.gamma
    ACTOR_LR = cfg.rl.actor_lr
    CRITIC_LR = cfg.rl.critic_lr
    J_STAR = 0.0

    # Buffer
    BETA_START = cfg.rl.buffer_beta
    BETA_FRAMES = MAX_EPISODES

    env = SatelliteEnv(
        max_steps=MAX_STEPS,
        max_sim_time=cfg.simulation.n_orbits_sim * T_orbital,
        tol=cfg.rl.integration_tol,
        initial_mass=cfg.satellite.initial_mass_kg,
        propellant_mass=cfg.satellite.propellant_mass_kg,
        Isp=cfg.satellite.isp_s,
        initial_state=cartesian_state,
        target_orbit=target_mee,
        max_delta_v=cfg.satellite.max_delta_v_mps,
        max_coast_fraction=cfg.satellite.max_coast_fraction,
        state_scales=cfg.rl.state_scales,
        termination_tol=cfg.rl.termination_tol,
        penalty_weights=cfg.rl.penalty_weights,
        delta_p_min=-150000.0,
        delta_p_max=150000.0,
        initial_keplerian=initial_keplerian,
    )

    actor = Actor(STATE_DIM, ACTION_DIM)
    critic = Critic(STATE_DIM, ACTION_DIM)

    target_actor = Actor(STATE_DIM, ACTION_DIM)
    target_critic = Critic(STATE_DIM, ACTION_DIM)

    # Buffer initialization
    buffer = PrioritizedReplayBuffer(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_size=cfg.rl.buffer_capacity,
        alpha=0.8,
    )

    # --- WARM-UP REPLAY BUFFER ---
    print(f"\n[Seed: {seed}] Filling the replay buffer with initial warm-up steps...")
    warmup_steps = cfg.rl.batch_size

    state, _ = env.reset(seed=seed)

    agent = Agent(
        action_dim=ACTION_DIM,
        exploration_noise=cfg.rl.exploration_noise,
        min_noise=cfg.rl.min_noise,
        noise_decay=cfg.rl.noise_decay,
    )

    for i in range(warmup_steps):
        # Uniform random action sampling during warm-up for unbiased buffer initialization
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        buffer.add(state, action, reward, next_state, terminated or truncated)

        if terminated or truncated:
            state, _ = env.reset(seed=seed)
        else:
            state = next_state

        if i % 10 == 0:
            print(f"[Seed: {seed}] Warm-up step {i}/{warmup_steps}")

    agent.exploration_noise = (
        cfg.rl.exploration_noise
    )  # Reset exploration noise after warm-up
    agent.min_noise = cfg.rl.min_noise
    agent.noise_decay = cfg.rl.noise_decay

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    print(f"[Seed: {seed}] Replay buffer warm-up completed.")
    # ---------------------------

    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    actor_scheduler = ReduceLROnPlateau(
        actor_optimizer, mode="min", factor=0.5, patience=50
    )

    print(f"[Seed: {seed}] Starting off-policy ADHDP with {MAX_STEPS} steps...")

    ROLLING_WINDOW = 50
    reward_window = deque(maxlen=ROLLING_WINDOW)
    best_avg_reward = -np.inf

    actor_best_path = os.path.join(save_dir, "actor_best.pth")
    critic_best_path = os.path.join(save_dir, "critic_best.pth")
    actor_final_path = os.path.join(save_dir, "actor_final.pth")

    rewards_history = []
    min_p_error_history = []
    actor_loss_history = []
    critic_loss_history = []

    initial_time = time.time()
    success_count = 0  # Count of episodes with positive reward

    total_episodes = 0  # Total episodes

    for episode in range(MAX_EPISODES):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        done = False
        step = 0
        episode_reward = 0.0

        min_ep_p_error = np.inf
        ep_actor_losses = []
        ep_critic_losses = []

        # if success_count >= 50:
        #     print(
        #         f"[Seed: {seed}] Ep {episode}: tightening termination tolerance "
        #         f"({env.termination_tol:.2e} -> {env.termination_tol*1e-1:.2e})"
        #     )
        #     env.termination_tol *= 1.0e-1
        #     success_count = 0
        #     if env.termination_tol < 1.0:
        #         print(f"[Seed: {seed}] Soglia minima raggiunta. Stop training.")
        #         break

        done = False
        step = 0
        episode_reward = 0.0

        min_ep_p_error = np.inf

        current_beta = min(1.0, BETA_START + episode * (1.0 - BETA_START) / BETA_FRAMES)
        # print(f"[Seed: {seed}] Episode {episode} |")

        while not done and step < MAX_STEPS:

            # One step of interaction with the environment, to be added to the replay buffer

            action = agent.select_action(state, actor, add_noise=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # print(
            #     f"{'Done' if done else 'Step'} {step} | Reward: {reward:.4f} | p_error: {abs(env.equinoctial_parameters[0] - env.target_orbit[0]):.2f} m"
            # )
            # if reward > 0.0:
            #     print(
            #         f"[Ep {episode}] Positive reward: {reward:.4f} at step {step} | "
            #         f"p_error: {abs(env.equinoctial_parameters[0] - env.target_orbit[0]):.2f} m"
            #     )
            #     reward = 0.0

            current_p_err = abs(env.equinoctial_parameters[0] - env.target_orbit[0])
            current_kep = element_conversion.mee_to_keplerian(
                env.equinoctial_parameters, False
            )
            current_e_err = abs(current_kep[1] - target_keplerian[1])
            current_i_err = abs(current_kep[2] - target_keplerian[2])
            if current_p_err < min_ep_p_error:
                min_ep_p_error = current_p_err

            # Add transition to the prioritized replay buffer
            # From here, we will sample a batch of transitions and perform the ADHDP update
            buffer.add(state, action, reward, next_state, done)
            (
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch,
                tree_indices,
                weights,
            ) = buffer.sample(batch_size=cfg.rl.batch_size, beta=current_beta)
            # reward_batch = reward_batch * 100

            current_J_batch = critic(state_batch, action_batch)

            # with torch.no_grad():
            #     next_action_batch = actor(next_state_batch)
            #     next_J_batch = critic(next_state_batch, next_action_batch)
            #     target_J_batch = reward_batch + GAMMA * next_J_batch * (1 - done_batch)

            with torch.no_grad():
                next_actions = target_actor(next_state_batch)
                next_J_batch = target_critic(next_state_batch, next_actions)
                target_J_batch = reward_batch + GAMMA * next_J_batch * (1 - done_batch)

            td_error_batch = current_J_batch - target_J_batch

            if torch.isnan(td_error_batch).any() or torch.isinf(td_error_batch).any():
                print("Warning: NaN or Inf detected in TD error. Skipping this update.")
                continue

            # Weighted loss for Importance Sampling
            critic_loss = 0.5 * (weights * (td_error_batch**2)).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()

            # Update priorities in Sum-Tree
            buffer.update_priorities(
                tree_indices, td_error_batch.detach().cpu().numpy()
            )

            for param in critic.parameters():
                param.requires_grad = False

            actor_action_train = actor(state_batch)
            predicted_J = critic(state_batch, actor_action_train)

            # e_a = predicted_J - J_STAR
            # actor_loss = 0.5 * (e_a**2).mean()

            actor_loss = -predicted_J.mean()  # invece di 0.5*(predicted_J - J_STAR)**2
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()

            ep_actor_losses.append(actor_loss.item())
            ep_critic_losses.append(critic_loss.item())

            for param in critic.parameters():
                param.requires_grad = True

            # --- SOFT UPDATE OF TARGET NETWORKS ---
            for param, target_param in zip(
                critic.parameters(), target_critic.parameters()
            ):
                target_param.data.copy_(
                    cfg.rl.tau * param.data + (1 - cfg.rl.tau) * target_param.data
                )

            for param, target_param in zip(
                actor.parameters(), target_actor.parameters()
            ):
                target_param.data.copy_(
                    cfg.rl.tau * param.data + (1 - cfg.rl.tau) * target_param.data
                )

            if done:
                print(
                    f"[Ep {episode}] critic_loss={critic_loss.item():.6f} | "
                    f"actor_loss={actor_loss.item():.6f} | "
                    f"predicted_J mean/min/max = "
                    f"{predicted_J.mean().item():.4f}/{predicted_J.min().item():.4f}/{predicted_J.max().item():.4f} | \n"
                    f"weights mean/min/max = "
                    f"{weights.mean().item():.4f}/{weights.min().item():.4f}/{weights.max().item():.4f} | "
                    f"Steps: {step} | "
                    f"{'Terminated'if terminated else 'Truncated'} | \n"
                    f"Target p error: {env.current_target_delta_p:.2f} m | "
                    f"min_p_error: {min_ep_p_error:.2f} m | "
                    f"final_e_error: {current_e_err:.4e} rad | "
                    f"final_i_error: {current_i_err:.4e} rad \n "
                )

            state = next_state
            episode_reward += float(reward)

            step += 1

        agent.exploration_noise = max(
            agent.min_noise, agent.exploration_noise * agent.noise_decay
        )

        if terminated > 0.0:
            success_count += 1

            print(
                f"[Ep {episode}] Positive reward: {reward:.4f} at step {step} | "
                f"p_error: {abs(env.equinoctial_parameters[0] - env.target_orbit[0]):.2f} m | "
                f"e_error: {abs(current_kep[1] - target_keplerian[1]):.4e} rad | "
                f"i_error: {abs(current_kep[2] - target_keplerian[2]):.4e} rad"
            )
            # episode_reward = 0.0
        else:
            success_count = 0

        # Scheduler update based on the minimum p error of the episode
        actor_scheduler.step(min_ep_p_error)
        rewards_history.append(episode_reward)
        min_p_error_history.append(min_ep_p_error)
        actor_loss_history.append(
            float(np.mean(ep_actor_losses)) if ep_actor_losses else 0.0
        )
        critic_loss_history.append(
            float(np.mean(ep_critic_losses)) if ep_critic_losses else 0.0
        )

        # Reward tracking for best model saving
        reward_window.append(episode_reward)

        if len(reward_window) == ROLLING_WINDOW:
            avg_reward = float(np.mean(reward_window))
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(actor.state_dict(), actor_best_path)
                torch.save(critic.state_dict(), critic_best_path)

        if episode % 50 == 0:
            current_time = time.time() - initial_time
            best_display = (
                best_avg_reward if best_avg_reward > -np.inf else float("nan")
            )
            print(
                f"[Seed: {seed}] Ep: {episode} | Reward: {episode_reward:.4f} | Noise: {agent.exploration_noise:.4f} | Best {ROLLING_WINDOW}-ep avg: {best_display:.4f} | Elapsed Time: {current_time:.2f}s"
            )

        if success_count >= 50:
            break  # Stop training if 50 consecutive successful episodes are achieved

    np.save(os.path.join(save_dir, "rewards_history.npy"), np.array(rewards_history))
    np.save(
        os.path.join(save_dir, "min_p_error_history.npy"), np.array(min_p_error_history)
    )
    np.save(
        os.path.join(save_dir, "actor_loss_history.npy"), np.array(actor_loss_history)
    )
    np.save(
        os.path.join(save_dir, "critic_loss_history.npy"), np.array(critic_loss_history)
    )

    torch.save(actor.state_dict(), actor_final_path)
    print(f"\n[Seed: {seed}] Training completed. Weights and logs saved in: {save_dir}")

    if best_avg_reward > -np.inf:
        eval_actor = Actor(STATE_DIM, ACTION_DIM)
        eval_actor.load_state_dict(torch.load(actor_best_path))
        print(f"[Seed: {seed}] Evaluating BEST checkpoint:")
        evaluate_policy(eval_actor, env, num_episodes=100, max_steps=MAX_STEPS)
    else:
        print(f"[Seed: {seed}] No optimal checkpoint saved, evaluating final weights:")
        evaluate_policy(actor, env, num_episodes=100, max_steps=MAX_STEPS)


def run_experiments(seeds: list, base_save_dir: str = None):
    if base_save_dir is None:
        base_save_dir = str(current_dir / "results")
    start_time = time.time()
    for s in seeds:
        train(seed=s, base_save_dir=base_save_dir)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # test_seeds = [42, 100, 2024]
    test_seeds = [2024]

    run_experiments(seeds=test_seeds)
