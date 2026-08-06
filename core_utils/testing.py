import os
import pathlib
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to sys.path
current_dir = pathlib.Path(__file__).parent.resolve()
parent_dir = current_dir.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from tudatpy.astro import element_conversion

# Matplotlib settings for publication-quality plots
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "Times New Roman",
            "DejaVu Serif",
            "serif",
        ],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 16,
    }
)


def plot_loss_curves(
    episodes,
    actor_loss_mean,
    actor_loss_std=None,
    critic_loss_mean=None,
    critic_loss_std=None,
    output_filename="adhdp_loss_curves.pdf",
    output_dir="./Plots/off_policy_test",
):
    """
    Plot 1: Actor and Critic Loss curves during training (2 subplots side-by-side).
    """
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Actor Loss Subplot
    ax1.plot(
        episodes, actor_loss_mean, color="tab:blue", linewidth=1.5, label="Actor Loss"
    )
    if actor_loss_std is not None and np.any(actor_loss_std > 0):
        ax1.fill_between(
            episodes,
            actor_loss_mean - actor_loss_std,
            actor_loss_mean + actor_loss_std,
            color="tab:blue",
            alpha=0.2,
        )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(r"Actor Loss $\mathcal{L}_{\mathrm{actor}}$")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Critic Loss Subplot
    if critic_loss_mean is not None:
        ax2.plot(
            episodes,
            critic_loss_mean,
            color="tab:red",
            linewidth=1.5,
            label="Critic Loss",
        )
        if critic_loss_std is not None and np.any(critic_loss_std > 0):
            ax2.fill_between(
                episodes,
                critic_loss_mean - critic_loss_std,
                critic_loss_mean + critic_loss_std,
                color="tab:red",
                alpha=0.2,
            )
        ax2.set_xlabel("Episode")
        ax2.set_ylabel(r"Critic Loss $\mathcal{L}_{\mathrm{critic}}$")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right")

    plt.tight_layout()
    filepath = out_path / output_filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot: {filepath}")


def plot_reward_curve(
    episodes,
    rewards_mean,
    rewards_std=None,
    output_filename="adhdp_reward_curve.pdf",
    output_dir="./Plots/off_policy_test",
):
    """
    Plot 4: Cumulative Reward curve over training episodes (logarithmic scale).
    Displays raw episode reward (Steel Blue background line), std band (multi-seed variance),
    and a prominent bold Crimson rolling mean trendline for maximum visual distinction.
    """
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    # Raw Episode Reward line (Steel Blue, semi-transparent)
    ax.plot(
        episodes,
        rewards_mean,
        color="#4682b4",  # Steel Blue
        alpha=0.45,
        linewidth=0.9,
        label="Raw Episode Reward",
    )

    # Multi-seed Standard Deviation Band (+/- std across seeds)
    if rewards_std is not None and np.any(rewards_std > 0):
        ax.fill_between(
            episodes,
            rewards_mean - rewards_std,
            rewards_mean + rewards_std,
            color="#a6bddb",
            alpha=0.3,
            label=r"Multi-seed Std Dev ($\pm\sigma$)",
        )

    # Rolling Mean Trendline (Bold Crimson / Burnt Orange)
    if len(rewards_mean) >= 20:
        window = 20
        rolling_mean = np.convolve(rewards_mean, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1 :],
            rolling_mean,
            color="#d95f02",  # High-contrast Burnt Orange/Crimson
            linewidth=2.4,
            linestyle="-",
            label=f"Rolling Mean ({window} ep)",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_yscale("symlog", linthresh=10.0)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    filepath = out_path / output_filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward plot (high contrast log scale): {filepath}")


def plot_orbit_errors(
    times,
    p_errors,
    a_errors,
    output_filename="adhdp_orbit_errors_p_a.pdf",
    output_dir="./Plots/off_policy_test",
):
    """
    Plot 2: Error in parameter p and semi-major axis a over time (2 subplots side-by-side, logarithmic scale).
    """
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    abs_p_errors = np.abs(p_errors)
    abs_a_errors = np.abs(a_errors)

    # Error in p (log scale)
    ax1.plot(
        times, abs_p_errors, color="tab:blue", linewidth=1.8, marker="o", markersize=3
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"Parameter $p$ Error $|\Delta p|$ [m]")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    # Error in semi-major axis a (log scale)
    ax2.plot(
        times, abs_a_errors, color="tab:orange", linewidth=1.8, marker="s", markersize=3
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel(r"Semi-Major Axis Error $|\Delta a|$ [m]")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    filepath = out_path / output_filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved orbit errors plot (log scale): {filepath}")


def plot_actions_and_coasting(
    step_times,
    delta_v_values,
    coasting_times,
    output_filename="adhdp_actions_deltav_coasting.pdf",
    output_dir="./Plots/off_policy_test",
):
    """
    Plot 3: Applied Delta V and Coasting Time over time (2 subplots side-by-side).
    """
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Subplot 1: Delta V
    ax1.step(
        step_times,
        delta_v_values,
        where="post",
        color="tab:purple",
        linewidth=1.8,
        marker="o",
        label=r"Applied $\Delta v$",
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"Applied $\Delta v$ [m/s]")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Subplot 2: Coasting Time
    ax2.step(
        step_times,
        coasting_times,
        where="post",
        color="tab:green",
        linewidth=1.8,
        marker="s",
        label=r"Coasting Time $t_{\mathrm{coast}}$ [s]",
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel(r"Coasting Time $t_{\mathrm{coast}}$ [s]")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    filepath = out_path / output_filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved actions & coasting plot: {filepath}")


def plot_keplerian_errors(
    times,
    e_errors,
    inc_errors,
    omega_errors,
    raan_errors,
    output_dir="Plots/off_policy_test",
):
    """
    Plots individual errors of Keplerian elements (Eccentricity, Inclination, Arg. of Periapsis, RAAN)
    both as 4 individual plot files and as a combined 2x2 multi-panel plot.
    """
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    # 1. Eccentricity Error Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, e_errors, color="#d95f02", linewidth=1.8, label=r"$\Delta e$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Eccentricity Error $\Delta e$ [-]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    p1 = out_path / "error_eccentricity.pdf"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Inclination Error Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, inc_errors, color="#7570b3", linewidth=1.8, label=r"$\Delta i$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Inclination Error $\Delta i$ [deg]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    p2 = out_path / "error_inclination.pdf"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Argument of Periapsis Error (omega) Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, omega_errors, color="#1b9e77", linewidth=1.8, label=r"$\Delta \omega$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Arg. of Periapsis Error $\Delta \omega$ [deg]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    p3 = out_path / "error_arg_periapsis.pdf"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. RAAN Error (Omega) Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, raan_errors, color="#e7298a", linewidth=1.8, label=r"$\Delta \Omega$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"RAAN Error $\Delta \Omega$ [deg]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    p4 = out_path / "error_raan.pdf"
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5. Combined 2x2 Multi-panel Overview Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    axes[0, 0].plot(times, e_errors, color="#d95f02", linewidth=1.6)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel(r"Eccentricity Error $\Delta e$ [-]")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].set_title("Eccentricity Error")

    axes[0, 1].plot(times, inc_errors, color="#7570b3", linewidth=1.6)
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel(r"Inclination Error $\Delta i$ [deg]")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].set_title("Inclination Error")

    axes[1, 0].plot(times, omega_errors, color="#1b9e77", linewidth=1.6)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel(r"Arg. of Periapsis Error $\Delta \omega$ [deg]")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].set_title("Argument of Periapsis Error")

    axes[1, 1].plot(times, raan_errors, color="#e7298a", linewidth=1.6)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel(r"RAAN Error $\Delta \Omega$ [deg]")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].set_title("RAAN Error")

    plt.tight_layout()
    p_comb = out_path / "keplerian_elements_errors.pdf"
    plt.savefig(p_comb, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Keplerian element error plots to: {out_path}")


@torch.no_grad()
def run_deterministic_test_episode(actor, env, target_delta_p=None):
    """
    Executes a deterministic test episode and extracts:
    - High resolution state trajectory times, p_errors, and a_errors
    - Keplerian element errors (e, inclination, arg of periapsis, RAAN)
    - Per-step applied Delta V and coasting time
    """
    actor.eval()
    if target_delta_p is not None:
        state, _ = env.reset(target_delta_p=target_delta_p)
    else:
        state, _ = env.reset()
    done = False

    target_p = env.target_orbit[0]
    target_f = env.target_orbit[1]
    target_g = env.target_orbit[2]
    target_ecc_sq = target_f**2 + target_g**2
    target_a = target_p / (1.0 - target_ecc_sq) if target_ecc_sq < 1.0 else target_p

    init_kep = element_conversion.cartesian_to_keplerian(env.cartesian_state, env.mu)
    target_e = np.sqrt(target_ecc_sq)
    target_inc = init_kep[2]
    target_omega = init_kep[3]
    target_raan = init_kep[4]

    times = []
    p_errors = []
    a_errors = []
    e_errors = []
    inc_errors = []
    omega_errors = []
    raan_errors = []

    step_times = []
    delta_v_values = []
    coasting_times = []

    current_t = 0.0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = actor(state_tensor)
        action = action_tensor.squeeze(0).numpy()

        thrust_cmd = action[0]
        applied_delta_v = thrust_cmd * env.max_delta_v

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        c_time = info.get("coasting_time", 0.0)
        step_times.append(current_t)
        delta_v_values.append(applied_delta_v)
        coasting_times.append(c_time)

        # High resolution propagation history
        prop_times = info.get("step_times", [])
        prop_states = info.get("step_states", [])

        if len(prop_times) > 0:
            for t, st in zip(prop_times, prop_states):
                mee = element_conversion.cartesian_to_mee(st[:6], env.mu)
                p = mee[0]
                f, g = mee[1], mee[2]
                ecc_sq = f**2 + g**2
                a = p / (1.0 - ecc_sq) if ecc_sq < 1.0 else p

                kep = element_conversion.cartesian_to_keplerian(st[:6], env.mu)
                e_val = kep[1]
                inc_val = kep[2]
                omega_val = kep[3]
                raan_val = kep[4]

                times.append(t)
                p_errors.append(p - target_p)
                a_errors.append(a - target_a)
                e_errors.append(e_val - target_e)
                inc_errors.append(np.degrees(inc_val - target_inc))
                omega_errors.append(np.degrees(omega_val - target_omega))
                raan_errors.append(np.degrees(raan_val - target_raan))
        else:
            mee = env.equinoctial_parameters
            p = mee[0]
            f, g = mee[1], mee[2]
            ecc_sq = f**2 + g**2
            a = p / (1.0 - ecc_sq) if ecc_sq < 1.0 else p
            kep = element_conversion.cartesian_to_keplerian(env.cartesian_state, env.mu)

            times.append(env.current_time)
            p_errors.append(p - target_p)
            a_errors.append(a - target_a)
            e_errors.append(kep[1] - target_e)
            inc_errors.append(np.degrees(kep[2] - target_inc))
            omega_errors.append(np.degrees(kep[3] - target_omega))
            raan_errors.append(np.degrees(kep[4] - target_raan))

        current_t = env.current_time
        state = next_state

    return (
        np.array(times),
        np.array(p_errors),
        np.array(a_errors),
        np.array(step_times),
        np.array(delta_v_values),
        np.array(coasting_times),
        np.array(e_errors),
        np.array(inc_errors),
        np.array(omega_errors),
        np.array(raan_errors),
    )
