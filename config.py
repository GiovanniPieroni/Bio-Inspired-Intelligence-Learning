from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np


@dataclass
class OrbitConfig:
    altitude_m: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_of_periapsis_deg: float
    true_anomaly_deg: float
    earth_radius_m: float
    mu_m3_s2: float

    def __post_init__(self):
        self.altitude_m = float(self.altitude_m)
        self.eccentricity = float(self.eccentricity)
        self.inclination_deg = float(self.inclination_deg)
        self.raan_deg = float(self.raan_deg)
        self.arg_of_periapsis_deg = float(self.arg_of_periapsis_deg)
        self.true_anomaly_deg = float(self.true_anomaly_deg)
        self.earth_radius_m = float(self.earth_radius_m)
        self.mu_m3_s2 = float(self.mu_m3_s2)

    # Helper properties for automatic unit conversions and orbital calculations
    @property
    def inclination_rad(self) -> float:
        return float(np.radians(self.inclination_deg))

    @property
    def raan_rad(self) -> float:
        return float(np.radians(self.raan_deg))

    @property
    def arg_of_periapsis_rad(self) -> float:
        return float(np.radians(self.arg_of_periapsis_deg))

    @property
    def true_anomaly_rad(self) -> float:
        return float(np.radians(self.true_anomaly_deg))

    @property
    def semi_major_axis(self) -> float:
        r_apogee = self.earth_radius_m + self.altitude_m * (1 + self.eccentricity)
        r_perigee = self.earth_radius_m + self.altitude_m * (1 - self.eccentricity)
        return float((r_apogee + r_perigee) / 2.0)

    @property
    def orbital_period(self) -> float:
        return float(2 * np.pi * np.sqrt(self.semi_major_axis**3 / self.mu_m3_s2))


@dataclass
class SatelliteConfig:
    initial_mass_kg: float
    propellant_mass_kg: float
    isp_s: float
    max_delta_v_mps: float
    max_coast_fraction: float

    def __post_init__(self):
        self.initial_mass_kg = float(self.initial_mass_kg)
        self.propellant_mass_kg = float(self.propellant_mass_kg)
        self.isp_s = float(self.isp_s)
        self.max_delta_v_mps = float(self.max_delta_v_mps)
        self.max_coast_fraction = float(self.max_coast_fraction)


@dataclass
class RLConfig:
    state_dim: int
    action_dim: int
    num_episodes: int
    batch_size: int
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    exploration_noise: float
    min_noise: float
    noise_decay: float
    buffer_capacity: int
    buffer_beta: float
    penalty_weights: list
    integration_tol: float
    state_scales: list
    termination_tol: float
    termination_tol_a: float
    termination_tol_e: float

    def __post_init__(self):
        self.state_dim = int(self.state_dim)
        self.action_dim = int(self.action_dim)
        self.num_episodes = int(self.num_episodes)
        self.batch_size = int(self.batch_size)
        self.gamma = float(self.gamma)
        self.tau = float(self.tau)
        self.actor_lr = float(self.actor_lr)
        self.critic_lr = float(self.critic_lr)
        self.exploration_noise = float(self.exploration_noise)
        self.min_noise = float(self.min_noise)
        self.noise_decay = float(self.noise_decay)
        self.buffer_capacity = int(self.buffer_capacity)
        self.buffer_beta = float(self.buffer_beta)
        self.penalty_weights = [float(w) for w in self.penalty_weights]
        self.integration_tol = float(self.integration_tol)
        self.state_scales = [float(s) for s in self.state_scales]
        self.termination_tol = float(self.termination_tol)
        self.termination_tol_a = float(self.termination_tol_a)
        self.termination_tol_e = float(self.termination_tol_e)


@dataclass
class SimulationConfig:
    steps: int
    n_orbits_sim: float

    def __post_init__(self):
        self.steps = int(self.steps)
        self.n_orbits_sim = float(self.n_orbits_sim)


@dataclass
class DebugConfig:
    verbose: bool
    save_plots: bool
    plots_dir: str
    actor_weights_path: str
    critic_weights_path: str


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        path = Path(config_path)
        if not path.exists():
            project_root = Path(__file__).parent / "config.yaml"
            if project_root.exists():
                path = project_root
            else:
                raise FileNotFoundError(
                    f"Configuration file not found at: {path.resolve()}"
                )

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        self.orbit = OrbitConfig(**data["orbit"])
        self.satellite = SatelliteConfig(**data["satellite"])
        self.rl = RLConfig(**data["rl"])
        self.simulation = SimulationConfig(**data["simulation"])

        self.debug = DebugConfig(**data["debug"])


def load_config(config_path: str = "config.yaml") -> Config:
    """Helper function to load configuration."""
    return Config(config_path)
