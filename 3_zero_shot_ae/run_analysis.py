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
from SatelliteEnv_ae import SatelliteEnv
from config import load_config
from tudatpy.astro import element_conversion
from testing import plot_loss_curves, plot_reward_curve

# Matplotlib styling for publication-quality figures
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 13,
        "font.size": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 15,
    }
)


@torch.no_grad()
def run_deterministic_ae_test(actor, env, target_delta_a, target_delta_e):
    """Executes a deterministic rollout for a specific (target_delta_a, target_delta_e) pair."""
    actor.eval()
    state, info = env.reset(
        target_delta_a=target_delta_a, target_delta_e=target_delta_e
    )
    done = False

    times = []
    a_errors = []
    e_errors = []
    f_vals = []
    g_vals = []
    hp_vals = []
    ha_vals = []

    step_times = []
    delta_v_values = []
    coasting_times = []
    propellant_masses = []
    rsw_actions = []

    current_t = 0.0
    target_a = env.target_keplerian[0]
    target_e = env.target_keplerian[1]

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = actor(state_tensor).squeeze(0).numpy()

        thrust_cmd = action[0:3]
        applied_dv = np.linalg.norm(thrust_cmd) * env.max_delta_v

        next_state, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        c_time = step_info.get("coasting_time", 0.0)
        step_times.append(current_t)
        delta_v_values.append(applied_dv)
        coasting_times.append(c_time)
        propellant_masses.append(env.propellant_mass)
        rsw_actions.append(thrust_cmd)

        prop_times = step_info.get("step_times", [])
        prop_states = step_info.get("step_states", [])

        if len(prop_times) > 0:
            for t, st in zip(prop_times, prop_states):
                mee = element_conversion.cartesian_to_mee(st[:6], env.mu)
                p = mee[0]
                f, g = mee[1], mee[2]
                ecc_sq = f**2 + g**2
                a_val = p / (1.0 - ecc_sq) if ecc_sq < 1.0 else p
                e_val = np.sqrt(ecc_sq)
                hp = (a_val * (1.0 - e_val) - env.earth_spherical_radius) / 1e3
                ha = (a_val * (1.0 + e_val) - env.earth_spherical_radius) / 1e3

                times.append(t)
                a_errors.append(a_val - target_a)
                e_errors.append(e_val - target_e)
                f_vals.append(f)
                g_vals.append(g)
                hp_vals.append(hp)
                ha_vals.append(ha)

        state = next_state
        current_t = env.current_time

    return (
        np.array(times),
        np.array(a_errors),
        np.array(e_errors),
        np.array(f_vals),
        np.array(g_vals),
        np.array(hp_vals),
        np.array(ha_vals),
        np.array(step_times),
        np.array(delta_v_values),
        np.array(coasting_times),
        np.array(propellant_masses),
        np.array(rsw_actions),
        bool(terminated),
    )


def run_zero_shot_ae_analysis(
    base_dir=None, output_dir=None, seeds=None, test_targets=None
):
    print("=" * 80)
    print("Starting Joint Zero-Shot Δa + Δe ADHDP Multi-Orbit Analysis & Testing...")
    print("=" * 80)

    if base_dir is None:
        base_dir = str(current_dir / "results")
    if output_dir is None:
        output_dir = str(current_dir / "Plots")

    os.makedirs(output_dir, exist_ok=True)

    cfg = load_config()
    mu = cfg.orbit.mu_m3_s2
    initial_r = cfg.orbit.semi_major_axis
    T_orbital = 2 * np.pi * np.sqrt(initial_r**3 / mu)

    # Extract tolerances directly from config
    tol_a = getattr(cfg.rl, "termination_tol_a", getattr(cfg.rl, "termination_tol", 1000.0))
    tol_e = getattr(cfg.rl, "termination_tol_e", 0.005)

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

    if test_targets is None:
        test_targets = [
            (-120000.0, -0.04),
            (-120000.0, +0.04),
            (-60000.0, -0.02),
            (-60000.0, +0.02),
            (+60000.0, -0.02),
            (+60000.0, +0.02),
            (+120000.0, -0.04),
            (+120000.0, +0.04),
        ]

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
        termination_tol_a=tol_a,
        termination_tol_e=tol_e,
        penalty_weights=cfg.rl.penalty_weights,
        delta_a_min=-150000.0,
        delta_a_max=150000.0,
        delta_e_min=-0.05,
        delta_e_max=0.05,
        initial_keplerian=initial_keplerian,
    )

    # 1. Plot Training curves across seeds
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

    # 2. Zero-Shot Evaluation across (Δa, Δe) target pairs
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
                f"\nEvaluating Zero-Shot Checkpoint from {seed_dir} across {len(test_targets)} Test Orbits..."
            )
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]

            actor = Actor(state_dim, action_dim)
            actor.load_state_dict(torch.load(actor_path))

            results_by_orbit = {}

            print("\n" + "=" * 90)
            print(
                f"{'Target Δa [km]':<15} | {'Target Δe [-]':<15} | {'Agent Δv [m/s]':<16} | {'Final a_err [m]':<15} | {'Final e_err [-]':<15} | {'Steps':<6} | {'Status':<9}"
            )
            print("=" * 90)

            for da, de in test_targets:
                (
                    times,
                    a_errors,
                    e_errors,
                    f_vals,
                    g_vals,
                    hp_vals,
                    ha_vals,
                    step_times,
                    delta_v_values,
                    coasting_times,
                    propellant_masses,
                    rsw_actions,
                    terminated,
                ) = run_deterministic_ae_test(
                    actor, env, target_delta_a=da, target_delta_e=de
                )

                dv_agent = float(np.sum(delta_v_values))
                final_a_err = float(abs(a_errors[-1])) if len(a_errors) > 0 else 0.0
                final_e_err = float(abs(e_errors[-1])) if len(e_errors) > 0 else 0.0
                steps_count = len(step_times)
                status_str = "Reached" if terminated else "Truncated"

                results_by_orbit[(da, de)] = {
                    "times": times,
                    "a_errors": a_errors,
                    "e_errors": e_errors,
                    "f_vals": f_vals,
                    "g_vals": g_vals,
                    "hp_vals": hp_vals,
                    "ha_vals": ha_vals,
                    "step_times": step_times,
                    "delta_v_values": delta_v_values,
                    "coasting_times": coasting_times,
                    "propellant_masses": propellant_masses,
                    "rsw_actions": rsw_actions,
                    "dv_agent": dv_agent,
                    "final_a_err": final_a_err,
                    "final_e_err": final_e_err,
                    "steps": steps_count,
                    "terminated": terminated,
                }

                print(
                    f"{da/1e3:+14.1f} km | {de:+14.4f} | {dv_agent:15.2f} | {final_a_err:14.2f} m | {final_e_err:14.4e} | {steps_count:5d} | {status_str:<9}"
                )

            print("=" * 90 + "\n")

            num_orbits = len(test_targets)
            cmap = plt.get_cmap("tab10", max(10, num_orbits))
            line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
            line_widths = [2.5, 1.8, 2.8, 1.5, 2.6, 2.0, 2.7, 1.6]

            # --- PLOT 1: Semi-Major Axis Error Trajectories ---
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            for idx, (da, de) in enumerate(test_targets):
                res = results_by_orbit[(da, de)]
                if len(res["times"]) > 0:
                    abs_a_err = np.clip(np.abs(res["a_errors"]), 1.0, None)
                    ax.plot(
                        res["times"],
                        abs_a_err,
                        label=f"Δa={da/1e3:+5.0f}km, Δe={de:+5.2f}",
                        color=cmap(idx % 10),
                        linestyle=line_styles[idx % len(line_styles)],
                        linewidth=line_widths[idx % len(line_widths)],
                    )
            ax.axhline(
                y=tol_a,
                color="red",
                linestyle="--",
                linewidth=1.8,
                label=f"Tolerance ({tol_a:.0f} m)",
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"Semi-Major Axis Error $|\Delta a|$ [m]")
            ax.set_yscale("log")
            ax.set_title("Zero-Shot Semi-Major Axis Error Trajectories (Log Scale)")
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
            plt.tight_layout()
            p_a_traj = Path(output_dir) / "zero_shot_a_error_trajectories.pdf"
            plt.savefig(p_a_traj, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- PLOT 2: Eccentricity Error Trajectories ---
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            for idx, (da, de) in enumerate(test_targets):
                res = results_by_orbit[(da, de)]
                if len(res["times"]) > 0:
                    abs_e_err = np.clip(np.abs(res["e_errors"]), 1e-6, None)
                    ax.plot(
                        res["times"],
                        abs_e_err,
                        label=f"Δa={da/1e3:+5.0f}km, Δe={de:+5.2f}",
                        color=cmap(idx % 10),
                        linestyle=line_styles[idx % len(line_styles)],
                        linewidth=line_widths[idx % len(line_widths)],
                    )
            ax.axhline(
                y=tol_e,
                color="red",
                linestyle="--",
                linewidth=1.8,
                label=f"Tolerance ({tol_e:.1e})",
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"Eccentricity Error $|\Delta e|$ [-]")
            ax.set_yscale("log")
            ax.set_title("Zero-Shot Eccentricity Error Trajectories (Log Scale)")
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
            plt.tight_layout()
            p_e_traj = Path(output_dir) / "zero_shot_e_error_trajectories.pdf"
            plt.savefig(p_e_traj, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- NEW PLOT 3: Perigee and Apogee Altitudes (h_p and h_a) ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
            for idx, (da, de) in enumerate(test_targets):
                res = results_by_orbit[(da, de)]
                if len(res["times"]) > 0:
                    label_str = f"Δa={da/1e3:+5.0f}km, Δe={de:+5.2f}"
                    ax1.plot(
                        res["times"],
                        res["ha_vals"],
                        label=label_str,
                        color=cmap(idx % 10),
                        linestyle=line_styles[idx % len(line_styles)],
                        linewidth=line_widths[idx % len(line_widths)],
                    )
                    ax2.plot(
                        res["times"],
                        res["hp_vals"],
                        label=label_str,
                        color=cmap(idx % 10),
                        linestyle=line_styles[idx % len(line_styles)],
                        linewidth=line_widths[idx % len(line_widths)],
                    )

            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel(r"Apogee Altitude $h_a$ [km]")
            ax1.set_title(r"Apogee Altitude Evolution ($h_a$)")
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend(loc="upper left", framealpha=0.9, fontsize=8)

            ax2.axhline(
                y=250.0,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Perigee Safety Boundary (250 km)",
            )
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel(r"Perigee Altitude $h_p$ [km]")
            ax2.set_title(r"Perigee Altitude Evolution ($h_p$)")
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(loc="upper right", framealpha=0.9, fontsize=8)

            plt.tight_layout()
            p_alt = Path(output_dir) / "zero_shot_perigee_apogee_altitudes.pdf"
            plt.savefig(p_alt, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- NEW PLOT 4: MEE Eccentricity Phase Plane Trajectories (f, g) ---
            fig, ax = plt.subplots(figsize=(8, 7))
            
            # Initial point (f0, g0)
            first_res = list(results_by_orbit.values())[0]
            if len(first_res["f_vals"]) > 0:
                f0, g0 = first_res["f_vals"][0], first_res["g_vals"][0]
                ax.plot(f0, g0, marker="*", color="green", markersize=14, label="Initial State $(f_0, g_0)$", zorder=5)

            for idx, (da, de) in enumerate(test_targets):
                res = results_by_orbit[(da, de)]
                if len(res["f_vals"]) > 0:
                    label_str = f"Δa={da/1e3:+5.0f}km, Δe={de:+5.2f}"
                    ax.plot(
                        res["f_vals"],
                        res["g_vals"],
                        label=label_str,
                        color=cmap(idx % 10),
                        linestyle=line_styles[idx % len(line_styles)],
                        linewidth=line_widths[idx % len(line_widths)],
                    )
                    # Mark target point
                    target_f = res["f_vals"][-1]
                    target_g = res["g_vals"][-1]
                    ax.plot(target_f, target_g, marker="o", color=cmap(idx % 10), markersize=6, zorder=4)

            ax.set_xlabel(r"MEE Element $f = e \cos(\varpi)$ [-]")
            ax.set_ylabel(r"MEE Element $g = e \sin(\varpi)$ [-]")
            ax.set_title(r"MEE Eccentricity Phase Plane Trajectories $(f, g)$")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
            plt.tight_layout()
            p_fg = Path(output_dir) / "zero_shot_fg_phase_plane.pdf"
            plt.savefig(p_fg, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- NEW PLOT 5: Propellant Consumption m_fuel(t) ---
            fig, ax = plt.subplots(figsize=(9, 5))
            for idx, (da, de) in enumerate(test_targets):
                res = results_by_orbit[(da, de)]
                if len(res["step_times"]) > 0:
                    label_str = f"Δa={da/1e3:+5.0f}km, Δe={de:+5.2f}"
                    ax.plot(
                        res["step_times"],
                        res["propellant_masses"],
                        label=label_str,
                        color=cmap(idx % 10),
                        linestyle=line_styles[idx % len(line_styles)],
                        linewidth=line_widths[idx % len(line_widths)],
                    )
            ax.axhline(
                y=0.0,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Empty Propellant Tank (0 kg)",
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"Remaining Propellant Mass $m_{\mathrm{fuel}}$ [kg]")
            ax.set_title("Remaining Propellant Mass Over Time Across Test Orbits")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
            plt.tight_layout()
            p_fuel = Path(output_dir) / "zero_shot_propellant_consumption.pdf"
            plt.savefig(p_fuel, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- NEW PLOT 6: RSW Thrust Vector Components breakdown ---
            # Plot thrust components (Radial, Along-track, Cross-track) for a representative target pair
            sample_target = (+120000.0, +0.04)
            if sample_target in results_by_orbit:
                res_sample = results_by_orbit[sample_target]
                if len(res_sample["step_times"]) > 0 and len(res_sample["rsw_actions"]) > 0:
                    fig, (ax_r, ax_s, ax_w) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
                    rsw_acts = res_sample["rsw_actions"]
                    st_times = res_sample["step_times"]

                    ax_r.step(st_times, rsw_acts[:, 0], where="post", color="tab:red", linewidth=1.5)
                    ax_r.set_ylabel(r"Radial $u_R$ [-]")
                    ax_r.set_title(r"RSW Thrust Vector Breakdown ($\Delta a = +120$ km, $\Delta e = +0.04$)")
                    ax_r.grid(True, linestyle=":", alpha=0.6)

                    ax_s.step(st_times, rsw_acts[:, 1], where="post", color="tab:blue", linewidth=1.5)
                    ax_s.set_ylabel(r"Along-Track $u_S$ [-]")
                    ax_s.grid(True, linestyle=":", alpha=0.6)

                    ax_w.step(st_times, rsw_acts[:, 2], where="post", color="tab:green", linewidth=1.5)
                    ax_w.set_xlabel("Time [s]")
                    ax_w.set_ylabel(r"Cross-Track $u_W$ [-]")
                    ax_w.grid(True, linestyle=":", alpha=0.6)

                    plt.tight_layout()
                    p_rsw = Path(output_dir) / "zero_shot_rsw_thrust_components.pdf"
                    plt.savefig(p_rsw, dpi=300, bbox_inches="tight")
                    plt.close(fig)

            # --- PLOT 7: Cumulative Delta V and Coasting Duration Profiles (4 Subplots: 2x2 Grid) ---
            fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

            for idx, (da, de) in enumerate(test_targets):
                res = results_by_orbit[(da, de)]
                if len(res["step_times"]) > 0:
                    label_str = f"Δa={da/1e3:+5.0f}km, Δe={de:+5.2f}"
                    row_idx = 0 if da < 0 else 1

                    # Cumulative Delta V (smooth staircase)
                    cum_dv = np.cumsum(res["delta_v_values"])

                    # Left Column: Cumulative Delta V
                    axes[row_idx, 0].plot(
                        res["step_times"],
                        cum_dv,
                        label=label_str,
                        color=cmap(idx % 10),
                        linestyle=line_styles[idx % len(line_styles)],
                        linewidth=line_widths[idx % len(line_widths)],
                    )

                    # Right Column: Coasting Time Duration per Step
                    axes[row_idx, 1].plot(
                        res["step_times"],
                        res["coasting_times"],
                        label=label_str,
                        color=cmap(idx % 10),
                        linestyle=line_styles[idx % len(line_styles)],
                        linewidth=line_widths[idx % len(line_widths)],
                        alpha=0.85,
                    )

            # Subplot Titles & Formatting
            axes[0, 0].set_title(r"Cumulative $\Delta v$ Profile ($\Delta a < 0$)")
            axes[0, 0].set_xlabel("Time [s]")
            axes[0, 0].set_ylabel(r"Cumulative $\Delta v$ [m/s]")
            axes[0, 0].grid(True, linestyle=":", alpha=0.6)
            axes[0, 0].legend(loc="upper left", framealpha=0.9, fontsize=8)

            axes[0, 1].set_title(r"Coasting Duration Profile ($t_{\mathrm{coast}}$) ($\Delta a < 0$)")
            axes[0, 1].set_xlabel("Time [s]")
            axes[0, 1].set_ylabel(r"Coasting Time $t_{\mathrm{coast}}$ [s]")
            axes[0, 1].grid(True, linestyle=":", alpha=0.6)
            axes[0, 1].legend(loc="upper right", framealpha=0.9, fontsize=8)

            axes[1, 0].set_title(r"Cumulative $\Delta v$ Profile ($\Delta a > 0$)")
            axes[1, 0].set_xlabel("Time [s]")
            axes[1, 0].set_ylabel(r"Cumulative $\Delta v$ [m/s]")
            axes[1, 0].grid(True, linestyle=":", alpha=0.6)
            axes[1, 0].legend(loc="upper left", framealpha=0.9, fontsize=8)

            axes[1, 1].set_title(r"Coasting Duration Profile ($t_{\mathrm{coast}}$) ($\Delta a > 0$)")
            axes[1, 1].set_xlabel("Time [s]")
            axes[1, 1].set_ylabel(r"Coasting Time $t_{\mathrm{coast}}$ [s]")
            axes[1, 1].grid(True, linestyle=":", alpha=0.6)
            axes[1, 1].legend(loc="upper right", framealpha=0.9, fontsize=8)

            plt.tight_layout()
            p_action_profiles = (
                Path(output_dir) / "zero_shot_deltav_and_coasting_profiles.pdf"
            )
            plt.savefig(p_action_profiles, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # --- PLOT 8: Total Delta V Expenditure Bar Chart ---
            fig, ax = plt.subplots(figsize=(9, 5))
            labels = [f"Δa={da/1e3:+0.0f}k\nΔe={de:+0.2f}" for da, de in test_targets]
            dv_vals = [
                results_by_orbit[(da, de)]["dv_agent"] for da, de in test_targets
            ]
            x_indices = np.arange(len(labels))

            ax.bar(
                x_indices,
                dv_vals,
                width=0.45,
                color="#2b8cbe",
                edgecolor="#08589e",
                alpha=0.85,
            )
            ax.set_xticks(x_indices)
            ax.set_xticklabels(labels, rotation=15, fontsize=9)
            ax.set_ylabel("Total Applied $\Delta v$ [m/s]")
            ax.set_title("Zero-Shot Agent $\Delta v$ Expenditure Across Test Orbits")
            ax.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            p_dv = Path(output_dir) / "zero_shot_deltav_expenditure.pdf"
            plt.savefig(p_dv, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # Only analyze the most recently modified valid checkpoint
            break


if __name__ == "__main__":
    run_zero_shot_ae_analysis()
