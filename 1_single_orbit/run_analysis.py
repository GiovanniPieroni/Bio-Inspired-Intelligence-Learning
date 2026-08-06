import os
import pathlib
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to sys.path
current_dir = pathlib.Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
core_utils_dir = project_root / "core_utils"

for d in [str(current_dir), str(project_root), str(core_utils_dir)]:
    if d not in sys.path:
        sys.path.append(d)

from Models.FFN.model import Actor, Critic
from SatelliteEnv_P_only import SatelliteEnv
from config import load_config
from tudatpy.astro import element_conversion

from testing import (
    plot_loss_curves,
    plot_reward_curve,
    plot_orbit_errors,
    plot_actions_and_coasting,
    plot_keplerian_errors,
    run_deterministic_test_episode,
)


def run_analysis(
    base_dir=None, output_dir=None, seeds=None
):
    """
    Testing and analysis routine for off-policy ADHDP satellite trajectory optimization.
    Generates all required plots:
    1. Actor and Critic Loss curves over training episodes.
    2. Errors in parameter p and semi-major axis a (side-by-side subplots) over time.
    3. Applied Delta V and Coasting Time (side-by-side subplots) over time.
    4. Cumulative Reward curve over training episodes.
    """
    print("=" * 70)
    print("Starting Off-Policy ADHDP Analysis & Testing Routine...")
    print("=" * 70)

    if output_dir is None:
        output_dir = str(current_dir / "Plots")

    if base_dir is None:
        if (current_dir / "results").exists():
            base_dir = str(current_dir / "results")
        elif (parent_dir / "results").exists():
            base_dir = str(parent_dir / "results")
        else:
            base_dir = str(current_dir / "results")

    print(f"Loading training results from: {base_dir}")
    print(f"Plots will be saved to: {output_dir}")

    # Auto-detect available seed directories if not specified
    if seeds is None:
        base_path = Path(base_dir)
        found_seed_dirs = sorted(list(base_path.glob("seed_*")))
        seeds = []
        for sdir in found_seed_dirs:
            if sdir.is_dir():
                try:
                    seed_num = int(sdir.name.split("seed_")[1])
                    seeds.append(seed_num)
                except ValueError:
                    pass
        if not seeds:
            print(f"Warning: No seed_* directories found in {base_dir}.")
        else:
            print(f"Auto-detected available seeds: {seeds}")

    # 1. Environment initialization (exact match with train.py)
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
            cfg.orbit.semi_major_axis + 100000.0,  # 100 km higher orbit
            cfg.orbit.eccentricity,
            cfg.orbit.inclination_rad,
            cfg.orbit.arg_of_periapsis_rad,
            cfg.orbit.raan_rad,
            cfg.orbit.true_anomaly_rad,
        ]
    )

    target_mee = element_conversion.keplerian_to_mee(target_keplerian)
    T_orbital = 2 * np.pi * np.sqrt(cfg.orbit.semi_major_axis**3 / cfg.orbit.mu_m3_s2)

    env = SatelliteEnv(
        max_steps=cfg.simulation.steps,
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
    )

    # 2. Multi-seed training history collection
    all_rewards = []
    all_actor_losses = []
    all_critic_losses = []

    for seed in seeds:
        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        rew_path = os.path.join(seed_dir, "rewards_history.npy")
        actor_loss_path = os.path.join(seed_dir, "actor_loss_history.npy")
        critic_loss_path = os.path.join(seed_dir, "critic_loss_history.npy")

        if os.path.exists(rew_path):
            all_rewards.append(np.load(rew_path))
        if os.path.exists(actor_loss_path):
            all_actor_losses.append(np.load(actor_loss_path))
        if os.path.exists(critic_loss_path):
            all_critic_losses.append(np.load(critic_loss_path))

    # --- PLOT 4: Reward over time / episodes ---
    if len(all_rewards) > 0:
        min_len = min(len(r) for r in all_rewards)
        truncated_rewards = [r[:min_len] for r in all_rewards]
        episodes = np.arange(min_len)
        rewards_mean = np.mean(truncated_rewards, axis=0)
        rewards_std = np.std(truncated_rewards, axis=0)
        print("\n[1/4] Generating Reward curve plot...")
        plot_reward_curve(
            episodes=episodes,
            rewards_mean=rewards_mean,
            rewards_std=rewards_std,
            output_filename="reward_over_time.pdf",
            output_dir=output_dir,
        )
    else:
        print("Warning: No rewards_history.npy files found. Skipping reward plot.")

    # --- PLOT 1: Actor and Critic Loss curves ---
    if len(all_actor_losses) > 0 and len(all_critic_losses) > 0:
        min_len = min(
            min(len(a) for a in all_actor_losses),
            min(len(c) for c in all_critic_losses),
        )
        truncated_actor_losses = [a[:min_len] for a in all_actor_losses]
        truncated_critic_losses = [c[:min_len] for c in all_critic_losses]
        episodes = np.arange(min_len)
        actor_loss_mean = np.mean(truncated_actor_losses, axis=0)
        actor_loss_std = np.std(truncated_actor_losses, axis=0)
        critic_loss_mean = np.mean(truncated_critic_losses, axis=0)
        critic_loss_std = np.std(truncated_critic_losses, axis=0)

        print("\n[2/4] Generating Actor & Critic Loss curves plot...")
        plot_loss_curves(
            episodes=episodes,
            actor_loss_mean=actor_loss_mean,
            actor_loss_std=actor_loss_std,
            critic_loss_mean=critic_loss_mean,
            critic_loss_std=critic_loss_std,
            output_filename="actor_critic_losses.pdf",
            output_dir=output_dir,
        )
    else:
        print("Warning: Loss history files not found. Skipping loss curves plot.")

    # 3. Deterministic evaluation for single episode trajectory plots
    best_checkpoint_found = False

    # Sort seeds by checkpoint modification time (most recent first)
    def get_ckpt_mtime(s):
        sdir = Path(base_dir) / f"seed_{s}"
        p = sdir / "actor_best.pth"
        if not p.exists():
            p = sdir / "actor_final.pth"
        return p.stat().st_mtime if p.exists() else 0

    sorted_seeds = sorted(seeds, key=get_ckpt_mtime, reverse=True)

    for seed in sorted_seeds:
        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        actor_path = os.path.join(seed_dir, "actor_best.pth")
        if not os.path.exists(actor_path):
            actor_path = os.path.join(seed_dir, "actor_final.pth")

        if os.path.exists(actor_path):
            print(
                f"\n[3/4 & 4/4] Evaluating deterministic episode using checkpoint from {seed_dir}..."
            )
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            state_dict = torch.load(actor_path)

            if "rbf.centers" in state_dict:
                from Models.RBF.model import Actor as RBFActor
                actor = RBFActor(state_dim, action_dim)
            else:
                from Models.FFN.model import Actor as FFNActor
                hidden_dim = state_dict["fc1.weight"].shape[0] if "fc1.weight" in state_dict else 256
                actor = FFNActor(state_dim, action_dim, hidden_dim=hidden_dim)

            actor.load_state_dict(state_dict)

            (
                times,
                p_errors,
                a_errors,
                step_times,
                delta_v_values,
                coasting_times,
                e_errors,
                inc_errors,
                omega_errors,
                raan_errors,
            ) = run_deterministic_test_episode(actor, env)

            # --- PLOT 2: State error in p and semi-major axis a (side-by-side) ---
            print("Generating p & semi-major axis a error plots...")
            plot_orbit_errors(
                times=times,
                p_errors=p_errors,
                a_errors=a_errors,
                output_filename="orbit_errors_p_a.pdf",
                output_dir=output_dir,
            )

            # --- PLOT 3: Delta V and Coasting Time (side-by-side) ---
            print("Generating Delta V and Coasting Time plots...")
            plot_actions_and_coasting(
                step_times=step_times,
                delta_v_values=delta_v_values,
                coasting_times=coasting_times,
                output_filename="actions_deltav_coasting.pdf",
                output_dir=output_dir,
            )

            # --- PLOT 4: Individual & Combined Keplerian Element Errors ---
            print("Generating Keplerian elements error plots (e, inc, omega, RAAN)...")
            plot_keplerian_errors(
                times=times,
                e_errors=e_errors,
                inc_errors=inc_errors,
                omega_errors=omega_errors,
                raan_errors=raan_errors,
                output_dir=output_dir,
            )

            # --- SUMMARY PRINT: Total Delta V & Coasting Metrics ---
            total_delta_v = np.sum(np.abs(delta_v_values))
            mean_coasting_time = np.mean(coasting_times) if len(coasting_times) > 0 else 0.0
            total_coasting_time = np.sum(coasting_times)
            total_steps = len(step_times)

            print("\n" + "=" * 70)
            print("DETERMINISTIC EVALUATION METRICS SUMMARY:")
            print("=" * 70)
            print(f"  * Total Delta V Used   : {total_delta_v:.4f} m/s")
            print(f"  * Mean Coasting Time   : {mean_coasting_time:.2f} s ({mean_coasting_time/60.0:.2f} min)")
            print(f"  * Total Trajectory Time: {total_coasting_time:.2f} s ({total_coasting_time/3600.0:.2f} hours)")
            print(f"  * Total Steps Executed : {total_steps}")
            print(f"  * Final p_error        : {p_errors[-1]:.2f} m")
            print("=" * 70 + "\n")

            best_checkpoint_found = True
            break

    if not best_checkpoint_found:
        print(
            "Warning: No actor checkpoint weights found for deterministic evaluation."
        )

    print("=" * 70)
    print(f"Testing & analysis routine complete. All plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_analysis()
