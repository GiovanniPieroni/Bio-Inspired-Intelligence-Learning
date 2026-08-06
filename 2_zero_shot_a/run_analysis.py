import os
import pathlib
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    run_deterministic_test_episode,
)

# Matplotlib styling
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 13,
        "font.size": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 15,
    }
)


def compute_hohmann_deltav(r1, delta_p, mu=3.986004418e14):
    """Computes theoretical analytical Hohmann Delta V for semi-major axis change delta_p."""
    r2 = r1 + delta_p
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)
    a_trans = (r1 + r2) / 2.0

    v_trans1 = np.sqrt(mu * (2.0 / r1 - 1.0 / a_trans))
    v_trans2 = np.sqrt(mu * (2.0 / r2 - 1.0 / a_trans))

    dv1 = abs(v_trans1 - v1)
    dv2 = abs(v2 - v_trans2)
    return dv1 + dv2


def compute_hohmann_transfer_time(r1, delta_p, mu=3.986004418e14):
    """Computes theoretical analytical Hohmann transfer duration (half period of transfer ellipse)."""
    r2 = r1 + delta_p
    a_trans = (r1 + r2) / 2.0
    return np.pi * np.sqrt(a_trans**3 / mu)


def run_zero_shot_analysis(base_dir=None, output_dir=None, seeds=None, test_deltas=None):
    print("=" * 70)
    print("Starting Zero-Shot Multi-Orbit ADHDP Analysis & Testing Routine...")
    print("=" * 70)

    if base_dir is None:
        base_dir = str(current_dir / "results")
    if output_dir is None:
        output_dir = str(current_dir / "Plots")

    os.makedirs(output_dir, exist_ok=True)

    if seeds is None:
        if os.path.exists(base_dir):
            seeds = []
            for item in os.listdir(base_dir):
                if item.startswith("seed_") and os.path.isdir(
                    os.path.join(base_dir, item)
                ):
                    try:
                        seeds.append(int(item.split("_")[1]))
                    except ValueError:
                        pass
            seeds.sort()
        if not seeds:
            seeds = [2024]

    if test_deltas is None:
        test_deltas = [
            -120000.0,
            -80000.0,
            -40000.0,
            -20000.0,
            20000.0,
            40000.0,
            80000.0,
            120000.0,
        ]

    cfg = load_config()
    mu = cfg.orbit.mu_m3_s2
    initial_r = cfg.orbit.semi_major_axis
    T_orbital = 2 * np.pi * np.sqrt(initial_r**3 / mu)

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
    cartesian_state = element_conversion.keplerian_to_cartesian(initial_keplerian, mu)
    target_mee_base = element_conversion.keplerian_to_mee(initial_keplerian)

    env = SatelliteEnv(
        max_steps=cfg.simulation.steps,
        max_sim_time=cfg.simulation.n_orbits_sim * T_orbital,
        tol=cfg.rl.integration_tol,
        initial_mass=cfg.satellite.initial_mass_kg,
        propellant_mass=cfg.satellite.propellant_mass_kg,
        Isp=cfg.satellite.isp_s,
        initial_state=cartesian_state,
        target_orbit=target_mee_base,
        max_delta_v=cfg.satellite.max_delta_v_mps,
        max_coast_fraction=cfg.satellite.max_coast_fraction,
        state_scales=cfg.rl.state_scales,
        termination_tol=cfg.rl.termination_tol,
        penalty_weights=cfg.rl.penalty_weights,
        delta_p_min=-150000.0,
        delta_p_max=150000.0,
        initial_keplerian=initial_keplerian,
    )

    # 1. Training curves (aggregate across seeds)
    all_rewards = []
    all_actor_losses = []
    all_critic_losses = []

    for seed in seeds:
        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        r_file = os.path.join(seed_dir, "rewards_history.npy")
        a_file = os.path.join(seed_dir, "actor_loss_history.npy")
        c_file = os.path.join(seed_dir, "critic_loss_history.npy")

        if os.path.exists(r_file):
            all_rewards.append(np.load(r_file))
        if os.path.exists(a_file):
            all_actor_losses.append(np.load(a_file))
        if os.path.exists(c_file):
            all_critic_losses.append(np.load(c_file))

    if len(all_rewards) > 0:
        min_len = min(len(r) for r in all_rewards)
        rewards_arr = np.array([r[:min_len] for r in all_rewards])
        episodes = np.arange(min_len)
        rewards_mean = np.mean(rewards_arr, axis=0)
        rewards_std = np.std(rewards_arr, axis=0) if len(all_rewards) > 1 else None
        plot_reward_curve(episodes, rewards_mean, rewards_std, output_dir=output_dir)

    if len(all_actor_losses) > 0 and len(all_critic_losses) > 0:
        min_len_a = min(len(a) for a in all_actor_losses)
        min_len_c = min(len(c) for c in all_critic_losses)
        min_len = min(min_len_a, min_len_c)
        episodes = np.arange(min_len)

        a_losses = np.array([a[:min_len] for a in all_actor_losses])
        c_losses = np.array([c[:min_len] for c in all_critic_losses])

        plot_loss_curves(
            episodes=episodes,
            actor_loss_mean=np.mean(a_losses, axis=0),
            actor_loss_std=(
                np.std(a_losses, axis=0) if len(all_actor_losses) > 1 else None
            ),
            critic_loss_mean=np.mean(c_losses, axis=0),
            critic_loss_std=(
                np.std(c_losses, axis=0) if len(all_critic_losses) > 1 else None
            ),
            output_dir=output_dir,
        )

    # 2. Zero-Shot Multi-Orbit Evaluation
    def get_ckpt_mtime(s):
        sdir = Path(base_dir) / f"seed_{s}"
        p = sdir / "actor_best.pth"
        if not p.exists():
            p = sdir / "actor_final.pth"
        return p.stat().st_mtime if p.exists() else 0

    sorted_seeds = sorted(seeds, key=get_ckpt_mtime, reverse=True)
    best_checkpoint_found = False

    for seed in sorted_seeds:
        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        actor_path = os.path.join(seed_dir, "actor_best.pth")
        if not os.path.exists(actor_path):
            actor_path = os.path.join(seed_dir, "actor_final.pth")

        if os.path.exists(actor_path):
            print(
                f"\nEvaluating Zero-Shot Checkpoint from {seed_dir} across {len(test_deltas)} Test Orbits..."
            )
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            state_dict = torch.load(actor_path)

            if "rbf.centers" in state_dict:
                from Models.RBF.model import Actor as RBFActor

                actor = RBFActor(state_dim, action_dim)
            else:
                from Models.FFN.model import Actor as FFNActor

                actor = FFNActor(state_dim, action_dim)
            actor.load_state_dict(state_dict)

            results_by_orbit = {}

            print("\n" + "=" * 75)
            print(
                f"{'Target Δp [km]':<15} | {'Hohmann Δv [m/s]':<18} | {'Agent Δv [m/s]':<16} | {'Final p_err [m]':<14} | {'Steps':<6}"
            )
            print("=" * 75)

            for dp in test_deltas:
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
                ) = run_deterministic_test_episode(actor, env, target_delta_p=dp)

                dv_agent = float(np.sum(np.abs(delta_v_values)))
                dv_hohmann = float(compute_hohmann_deltav(initial_r, dp, mu=mu))
                time_hohmann = float(compute_hohmann_transfer_time(initial_r, dp, mu=mu))
                p_err_final = float(p_errors[-1])
                steps_count = len(step_times)

                results_by_orbit[dp] = {
                    "times": times,
                    "p_errors": p_errors,
                    "a_errors": a_errors,
                    "e_errors": e_errors,
                    "inc_errors": inc_errors,
                    "omega_errors": omega_errors,
                    "raan_errors": raan_errors,
                    "delta_v_values": delta_v_values,
                    "coasting_times": coasting_times,
                    "dv_agent": dv_agent,
                    "dv_hohmann": dv_hohmann,
                    "time_hohmann": time_hohmann,
                    "p_err_final": p_err_final,
                    "steps": steps_count,
                }

                print(
                    f"{dp/1e3:+14.1f} km | {dv_hohmann:17.2f} | {dv_agent:15.2f} | {p_err_final:13.1f} | {steps_count:5d}"
                )

            print("=" * 75 + "\n")

            # --- PLOT 1: Zero-Shot Orbit Error Trajectories (Modular Palette) ---
            fig, ax = plt.subplots(figsize=(9, 5.5))
            num_orbits = len(test_deltas)
            cmap = plt.get_cmap("tab10", max(10, num_orbits))
            
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                ax.plot(
                    res["times"],
                    res["p_errors"],
                    label=f"Target Δp = {dp/1e3:+6.1f} km",
                    color=cmap(idx % 10),
                    linewidth=1.8,
                )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Parameter $p$ Error [m]")
            ax.set_title("Zero-Shot Parameter $p$ Error Trajectories Across Test Orbits")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9)
            plt.tight_layout()
            p_traj = Path(output_dir) / "zero_shot_orbit_errors.pdf"
            plt.savefig(p_traj, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- PLOT 2: Modular Delta V Comparison (Agent vs Hohmann) ---
            fig, ax = plt.subplots(figsize=(9, 5))
            x_vals = [dp / 1e3 for dp in test_deltas]
            dv_ag = [results_by_orbit[dp]["dv_agent"] for dp in test_deltas]
            dv_hoh = [results_by_orbit[dp]["dv_hohmann"] for dp in test_deltas]

            # Dynamic bar width
            if len(x_vals) > 1:
                min_diff = np.min(np.diff(sorted(x_vals)))
                bar_width = max(1.5, min_diff * 0.35)
            else:
                bar_width = 4.0

            x_indices = np.arange(len(x_vals))
            ax.bar(
                x_indices - 0.175,
                dv_hoh,
                width=0.35,
                label="Theoretical Hohmann Δv",
                color="#a6bddb",
                edgecolor="#2b8cbe",
            )
            ax.bar(
                x_indices + 0.175,
                dv_ag,
                width=0.35,
                label="Zero-Shot Agent Δv",
                color="#d95f02",
                alpha=0.85,
            )

            ax.set_xticks(x_indices)
            ax.set_xticklabels([f"{x:+5.1f} km" for x in x_vals], rotation=15)
            ax.set_xlabel("Target Semi-Major Axis Change Δp [km]")
            ax.set_ylabel("Total Delta V [m/s]")
            ax.set_title("Zero-Shot Delta V Comparison: Agent vs Hohmann Limit")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left")
            plt.tight_layout()
            p_dv = Path(output_dir) / "zero_shot_deltav_comparison.pdf"
            plt.savefig(p_dv, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- PLOT 3: Zero-Shot Eccentricity Error Trajectories (Log Scale) ---
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                e_vals = np.abs(np.array(res["e_errors"]))
                e_vals_clipped = np.clip(e_vals, 1e-7, None)
                ax.plot(
                    res["times"],
                    e_vals_clipped,
                    label=f"Target Δp = {dp/1e3:+6.1f} km",
                    color=cmap(idx % 10),
                    linewidth=1.8,
                )
            ax.set_yscale("log")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"Eccentricity Error $\Delta e$ [-]")
            ax.set_title("Zero-Shot Eccentricity Error Trajectories Across Test Orbits (Log Scale)")
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9)
            plt.tight_layout()
            p_ecc = Path(output_dir) / "zero_shot_eccentricity_errors.pdf"
            plt.savefig(p_ecc, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- PLOT 4: SIDE-BY-SIDE Delta V & Coasting Time Bar Comparison ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
            x_vals = [dp / 1e3 for dp in test_deltas]
            dv_ag = [results_by_orbit[dp]["dv_agent"] for dp in test_deltas]
            dv_hoh = [results_by_orbit[dp]["dv_hohmann"] for dp in test_deltas]
            total_coasting = [
                sum(results_by_orbit[dp]["coasting_times"]) for dp in test_deltas
            ]

            x_indices = np.arange(len(x_vals))

            # Left Panel: Delta V
            ax1.bar(
                x_indices - 0.175,
                dv_hoh,
                width=0.35,
                label="Hohmann Δv Limit",
                color="#a6bddb",
                edgecolor="#2b8cbe",
            )
            ax1.bar(
                x_indices + 0.175,
                dv_ag,
                width=0.35,
                label="Zero-Shot Agent Δv",
                color="#d95f02",
                alpha=0.85,
            )
            ax1.set_xticks(x_indices)
            ax1.set_xticklabels([f"{x:+5.1f} km" for x in x_vals], rotation=25)
            ax1.set_xlabel("Target Semi-Major Axis Change Δp [km]")
            ax1.set_ylabel("Total Delta V [m/s]")
            ax1.set_title("(a) Total Applied Delta V vs Hohmann Limit")
            ax1.grid(True, linestyle=":", alpha=0.5)
            ax1.legend(loc="upper left")

            # Right Panel: Coasting Time vs Hohmann Transfer Time
            time_hoh = [results_by_orbit[dp]["time_hohmann"] for dp in test_deltas]
            ax2.bar(
                x_indices - 0.175,
                time_hoh,
                width=0.35,
                label="Hohmann Transfer Time",
                color="#a6bddb",
                edgecolor="#2b8cbe",
            )
            ax2.bar(
                x_indices + 0.175,
                total_coasting,
                width=0.35,
                color="#31a354",
                alpha=0.85,
                edgecolor="#006d2c",
                label="Agent Total Coasting Time",
            )
            ax2.set_xticks(x_indices)
            ax2.set_xticklabels([f"{x:+5.1f} km" for x in x_vals], rotation=25)
            ax2.set_xlabel("Target Semi-Major Axis Change Δp [km]")
            ax2.set_ylabel("Time [s]")
            ax2.set_title("(b) Transfer Time: Agent Coasting vs Hohmann Limit")
            ax2.grid(True, linestyle=":", alpha=0.5)
            ax2.legend(loc="upper left")

            plt.tight_layout()
            p_side_by_side_summary = Path(output_dir) / "zero_shot_deltav_coasting_side_by_side.pdf"
            plt.savefig(p_side_by_side_summary, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- PLOT 5: SIDE-BY-SIDE Cumulative Delta V & Cumulative Coasting Trajectories ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                step_dv = np.abs(res["delta_v_values"])
                cum_dv = np.cumsum(step_dv)
                cum_coast = np.cumsum(res["coasting_times"])
                # Step times matching the action updates
                steps_t = res["times"][: len(cum_dv)] if len(res["times"]) >= len(cum_dv) else np.arange(len(cum_dv)) * (T_orbital * 0.1)

                ax1.plot(
                    steps_t,
                    cum_dv,
                    label=f"Δp = {dp/1e3:+5.1f} km",
                    color=cmap(idx % 10),
                    linewidth=1.8,
                )
                ax2.plot(
                    steps_t,
                    cum_coast,
                    label=f"Δp = {dp/1e3:+5.1f} km",
                    color=cmap(idx % 10),
                    linewidth=1.8,
                )

            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Cumulative Delta V [m/s]")
            ax1.set_title("(a) Cumulative Applied Delta V Trajectories")
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend(loc="upper left", framealpha=0.85)

            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Cumulative Coasting Time [s]")
            ax2.set_title("(b) Cumulative Coasting Time Trajectories")
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(loc="upper left", framealpha=0.85)

            # --- PLOT 6: Individual Plots for Inclination, Omega, RAAN ---
            # Inclination Error Plot
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                ax.plot(
                    res["times"],
                    res["inc_errors"],
                    label=f"Target Δp = {dp/1e3:+6.1f} km",
                    color=cmap(idx % 10),
                    linewidth=1.8,
                )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"Inclination Error $\Delta i$ [deg]")
            ax.set_title("Zero-Shot Inclination Error Trajectories Across Test Orbits")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / "zero_shot_inclination_errors.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Argument of Periapsis Error Plot
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                ax.plot(
                    res["times"],
                    res["omega_errors"],
                    label=f"Target Δp = {dp/1e3:+6.1f} km",
                    color=cmap(idx % 10),
                    linewidth=1.8,
                )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"Arg. of Periapsis Error $\Delta \omega$ [deg]")
            ax.set_title("Zero-Shot Argument of Periapsis Error Trajectories")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / "zero_shot_omega_errors.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # RAAN Error Plot
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                ax.plot(
                    res["times"],
                    res["raan_errors"],
                    label=f"Target Δp = {dp/1e3:+6.1f} km",
                    color=cmap(idx % 10),
                    linewidth=1.8,
                )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"RAAN Error $\Delta \Omega$ [deg]")
            ax.set_title("Zero-Shot RAAN Error Trajectories Across Test Orbits")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / "zero_shot_raan_errors.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- PLOT 7: OTHER KEPLERIAN ELEMENT ERRORS COMBINED IN A 2x2 MULTI-PANEL FIGURE ---
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes_flat = axes.flatten()

            # Panel (a): Eccentricity e
            ax = axes_flat[0]
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                ax.plot(res["times"], res["e_errors"], color=cmap(idx % 10), label=f"Δp = {dp/1e3:+5.1f} km", linewidth=1.5)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Eccentricity Error $\Delta e$ [-]")
            ax.set_title("(a) Eccentricity Error $\Delta e$")
            ax.grid(True, linestyle=":", alpha=0.5)

            # Panel (b): Inclination i
            ax = axes_flat[1]
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                ax.plot(res["times"], res["inc_errors"], color=cmap(idx % 10), label=f"Δp = {dp/1e3:+5.1f} km", linewidth=1.5)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Inclination Error $\Delta i$ [deg]")
            ax.set_title("(b) Inclination Error $\Delta i$")
            ax.grid(True, linestyle=":", alpha=0.5)

            # Panel (c): Argument of Periapsis omega
            ax = axes_flat[2]
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                ax.plot(res["times"], res["omega_errors"], color=cmap(idx % 10), label=f"Δp = {dp/1e3:+5.1f} km", linewidth=1.5)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Arg. of Periapsis Error $\Delta \omega$ [deg]")
            ax.set_title("(c) Argument of Periapsis Error $\Delta \omega$")
            ax.grid(True, linestyle=":", alpha=0.5)

            # Panel (d): RAAN
            ax = axes_flat[3]
            for idx, dp in enumerate(test_deltas):
                res = results_by_orbit[dp]
                ax.plot(res["times"], res["raan_errors"], color=cmap(idx % 10), label=f"Δp = {dp/1e3:+5.1f} km", linewidth=1.5)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("RAAN Error $\Delta \Omega$ [deg]")
            ax.set_title("(d) RAAN Error $\Delta \Omega$")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

            plt.tight_layout()
            p_combined_all = Path(output_dir) / "zero_shot_all_keplerian_errors_combined.pdf"
            plt.savefig(p_combined_all, dpi=300, bbox_inches="tight")
            plt.close(fig)

            best_checkpoint_found = True
            break

    if not best_checkpoint_found:
        print(
            "Warning: No actor checkpoint weights found for deterministic evaluation."
        )

    print(
        f"Zero-Shot testing & analysis routine complete. All plots saved to: {output_dir}"
    )


if __name__ == "__main__":
    run_zero_shot_analysis()
