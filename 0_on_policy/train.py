import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

from agent import Agent
from Models.RBF.model import Actor, Critic
from SatelliteEnv_P_only import SatelliteEnv
from tudatpy.astro import element_conversion
from config import load_config
from torch.optim.lr_scheduler import ReduceLROnPlateau

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


def train(seed: int, base_save_dir: str = "results"):
    set_seed(seed)

    save_dir = os.path.join(base_save_dir, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    STATE_DIM = 8
    ACTION_DIM = 2
    MAX_EPISODES = 500
    MAX_STEPS = 2

    GAMMA = 0.98
    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-3
    J_STAR = 0.0

    env = SatelliteEnv(
        max_steps=MAX_STEPS,
        max_sim_time=1.5 * T_orbital,
        tol=1e-8,
        initial_mass=500.0,
        propellant_mass=100.0,
        Isp=300.0,
        initial_state=cartesian_state,
        target_orbit=target_mee,
        max_delta_v=656 / 2.0,
        max_coast_fraction=T_orbital / 2,
        state_scales=[6e5, 5.5e-2, 5.5e-2, 2.5e-2, 2.5e-2],
        termination_tol=100,
    )

    actor = Actor(STATE_DIM, ACTION_DIM)
    critic = Critic(STATE_DIM, ACTION_DIM)

    # --- WARM-UP RBF CENTERS ---
    print(
        f"\n[Seed: {seed}] Raccogliendo batch esplorativo per inizializzazione data-driven RBF..."
    )
    num_centers = actor.rbf.num_centers
    warmup_states = []
    warmup_actions = []

    state, _ = env.reset(seed=seed)

    agent = Agent(
        action_dim=ACTION_DIM,
        exploration_noise=0.25,
        min_noise=0.01,
        noise_decay=0.995,
    )

    for _ in range(num_centers):
        # Usa l'Actor con rumore per mappare il sottomominio di partenza reale
        action = agent.select_action(state, actor, add_noise=True)

        warmup_states.append(state)
        warmup_actions.append(action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    # Inizializzazione RBF
    warmup_states_tensor = torch.FloatTensor(np.array(warmup_states))
    warmup_actions_tensor = torch.FloatTensor(np.array(warmup_actions))

    actor.rbf.init_from_data(warmup_states_tensor)
    critic.rbf.init_from_data(
        torch.cat([warmup_states_tensor, warmup_actions_tensor], dim=1)
    )
    print(f"[Seed: {seed}] Inizializzazione RBF completata.")
    # ---------------------------

    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    actor_scheduler = ReduceLROnPlateau(
        actor_optimizer, mode="min", factor=0.5, patience=10
    )

    print(f"[Seed: {seed}] Iniziando ADHDP puro online a {MAX_STEPS} step...")

    ROLLING_WINDOW = 100
    reward_window = deque(maxlen=ROLLING_WINDOW)
    best_avg_reward = -np.inf

    actor_best_path = os.path.join(save_dir, "actor_best.pth")
    critic_best_path = os.path.join(save_dir, "critic_best.pth")
    actor_final_path = os.path.join(save_dir, "actor_final.pth")

    rewards_history = []
    min_p_error_history = []

    for episode in range(MAX_EPISODES):
        state, _ = env.reset(seed=seed if episode == 0 else None)
        done = False
        step = 0
        episode_reward = 0.0

        min_ep_p_error = np.inf

        while not done and step < MAX_STEPS:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            action = agent.select_action(state, actor, add_noise=True)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # --- TRACCIAMENTO ERRORE DIMENSIONALE ---
            current_p_err = abs(env.equinoctial_parameters[0] - env.target_orbit[0])
            if current_p_err < min_ep_p_error:
                min_ep_p_error = current_p_err

            # --- VALUTAZIONE CRITIC (ADHDP PURO) ---
            current_J = critic(state_tensor, action_tensor)

            with torch.no_grad():
                next_action = torch.FloatTensor(
                    agent.select_action(next_state, actor, add_noise=False)
                ).unsqueeze(0)
                # Si utilizza la STESSA rete critic per valutare il J futuro
                next_J = critic(next_state_tensor, next_action)
                target_J = reward + GAMMA * next_J * (1 - int(done))

            # Aggiornamento Critic
            e_c = current_J - target_J
            critic_loss = 0.5 * (e_c**2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()

            # --- AGGIORNAMENTO ACTOR (ADHDP PURO) ---
            for param in critic.parameters():
                param.requires_grad = False

            actor_action_train = actor(state_tensor)
            predicted_J = critic(state_tensor, actor_action_train)

            e_a = predicted_J - J_STAR
            actor_loss = 0.5 * (e_a**2).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()

            for param in critic.parameters():
                param.requires_grad = True

            state = next_state
            episode_reward += float(reward)
            step += 1

        actor_scheduler.step(min_ep_p_error)
        rewards_history.append(episode_reward)
        min_p_error_history.append(min_ep_p_error)

        reward_window.append(episode_reward)

        if len(reward_window) == ROLLING_WINDOW:
            avg_reward = float(np.mean(reward_window))
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(actor.state_dict(), actor_best_path)
                torch.save(critic.state_dict(), critic_best_path)

        if episode % 50 == 0:
            best_display = (
                best_avg_reward if best_avg_reward > -np.inf else float("nan")
            )
            print(
                f"[Seed: {seed}] Ep: {episode} | Reward: {episode_reward:.4f} | Noise: {agent.exploration_noise:.4f} | Best {ROLLING_WINDOW}-ep avg: {best_display:.4f}"
            )

    np.save(os.path.join(save_dir, "rewards_history.npy"), np.array(rewards_history))
    np.save(
        os.path.join(save_dir, "min_p_error_history.npy"), np.array(min_p_error_history)
    )

    torch.save(actor.state_dict(), actor_final_path)
    print(f"\n[Seed: {seed}] Training completato. Pesi e log salvati in: {save_dir}")

    if best_avg_reward > -np.inf:
        eval_actor = Actor(STATE_DIM, ACTION_DIM)
        eval_actor.load_state_dict(torch.load(actor_best_path))
        print(f"[Seed: {seed}] Evaluating BEST checkpoint:")
        evaluate_policy(eval_actor, env, num_episodes=100, max_steps=MAX_STEPS)
    else:
        print(
            f"[Seed: {seed}] Nessun checkpoint ottimo completato, valutando pesi finali:"
        )
        evaluate_policy(actor, env, num_episodes=100, max_steps=MAX_STEPS)


def run_experiments(seeds: list, base_save_dir: str = "results"):
    for s in seeds:
        train(seed=s, base_save_dir=base_save_dir)


if __name__ == "__main__":
    test_seeds = [42, 100, 2024]
    run_experiments(seeds=test_seeds)
