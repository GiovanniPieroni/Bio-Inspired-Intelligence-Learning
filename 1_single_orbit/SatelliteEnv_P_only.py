import numpy as np
import gymnasium as gym
import gc

from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
import tudatpy.astro.element_conversion as conversion
import tudatpy.astro.frame_conversion as frame_conversion

from utils import get_propellant_mass


class SatelliteEnv(gym.Env):
    def __init__(
        self,
        max_steps: int,
        max_sim_time: float,
        tol: float,
        initial_mass: float,
        propellant_mass: float,
        Isp: float,
        initial_state: np.ndarray,
        target_orbit: np.ndarray,
        max_delta_v: float,
        max_coast_fraction: float = 0.5,
        state_scales: list = [6e5, 5.5e-2, 5.5e-2, 2.5e-2, 2.5e-2],
        termination_tol: float = 1e-2,
        penalty_weights: list = [1.0, 0.0],
    ):
        super(SatelliteEnv, self).__init__()

        self.max_steps = max_steps
        self.max_sim_time = max_sim_time
        self.tol = tol
        self.initial_total_mass = initial_mass
        self.Isp = Isp
        self.initial_propellant_mass = propellant_mass
        self.initial_state = initial_state
        self.target_orbit = target_orbit
        self.max_delta_v = max_delta_v

        self.max_coast_fraction = max_coast_fraction
        self.state_scales = np.array(state_scales, dtype=np.float32)
        self.termination_tol = termination_tol
        self.penalty_weights = np.array(penalty_weights, dtype=np.float32)
        # Azione 2D: [Thrust Tangenziale, Frazione di Coasting]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Stato 8D: [p_err, f_err, g_err, h_err, k_err, sin(L), cos(L), mass_norm]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        spice.load_standard_kernels()
        bodies_to_create = ["Earth"]
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation
        )
        body_settings.add_empty_settings("Satellite")

        self.bodies = environment_setup.create_system_of_bodies(body_settings)
        self.mu = self.bodies.get(global_frame_origin).gravitational_parameter

        body_settings.get("Earth").shape_settings = (
            environment_setup.shape.spherical_spice()
        )
        self.earth_spherical_radius = self.bodies.get(
            "Earth"
        ).shape_model.average_radius

        self.bodies_to_propagate = ["Satellite"]
        self.central_bodies = ["Earth"]

        acceleration_settings_on_vehicle = dict(
            Earth=[propagation_setup.acceleration.point_mass_gravity()]
        )
        acceleration_settings = {"Satellite": acceleration_settings_on_vehicle}

        self.acceleration_models = propagation_setup.create_acceleration_models(
            self.bodies,
            acceleration_settings,
            self.bodies_to_propagate,
            self.central_bodies,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self.steps = 0
        self.cartesian_state = self.initial_state.copy()

        self.equinoctial_parameters = conversion.cartesian_to_mee(
            self.cartesian_state, self.mu
        )
        self.total_mass = self.initial_total_mass
        self.propellant_mass = self.initial_propellant_mass

        self.normalized_rl_state, self.raw_errors = (
            self._get_normalized_observation_and_error()
        )
        self.previous_state_cost = np.sum((self.raw_errors / self.state_scales) ** 2)

        return self.normalized_rl_state.copy(), {}

    def step(self, action):
        action_array = np.clip(
            np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0
        )
        thrust_action = action_array[0]
        coast_action = action_array[1]

        thrust_vector_s = thrust_action * self.max_delta_v
        thrust_vector_rsw = np.array([0.0, thrust_vector_s, 0.0], dtype=np.float64)

        rsw_to_inertial_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(
            self.cartesian_state
        )
        thrust_vector_inertial = rsw_to_inertial_matrix @ thrust_vector_rsw.T

        # Calcolo Dinamico del Coasting Time
        p, f, g = self.equinoctial_parameters[:3]
        ecc_sq = f**2 + g**2
        a = p / (1.0 - ecc_sq) if ecc_sq < 1.0 else p
        current_T_orbital = 2 * np.pi * np.sqrt(max(1.0, a) ** 3 / self.mu)

        # Mappa [-1, 1] -> [0, max_coast_fraction * current_T_orbital]
        min_coast_time = 1.0
        max_coast_time = self.max_coast_fraction * current_T_orbital
        coasting_time = min_coast_time + ((coast_action + 1.0) / 2.0) * (
            max_coast_time - min_coast_time
        )

        self.propellant_mass -= get_propellant_mass(
            np.linalg.norm(thrust_vector_inertial), self.Isp, self.total_mass
        )
        self.total_mass = self.initial_total_mass - (
            self.initial_propellant_mass - self.propellant_mass
        )

        state_history = self._run_propagation(thrust_vector_inertial, coasting_time)

        self.cartesian_state = list(state_history.values())[-1][:6]
        self.current_time = list(state_history.keys())[-1]

        self.equinoctial_parameters = conversion.cartesian_to_mee(
            self.cartesian_state, self.mu
        )
        self.normalized_rl_state, self.raw_errors = (
            self._get_normalized_observation_and_error()
        )

        current_state_cost = np.sum((self.raw_errors / self.state_scales) ** 2)
        error_reduction = self.previous_state_cost - current_state_cost
        self.previous_state_cost = current_state_cost

        # 1. Potential-Based Progress Reward (Reward for error reduction)
        progress_reward = 10.0 * error_reduction

        # 2. Soft MEE Perturbation Penalty (Penalizes deviations in f, g, h, k)
        mee_perturbation_penalty = -2.0 * np.sum(
            (self.raw_errors[1:5] / self.state_scales[1:5]) ** 2
        )

        # 3. Step Penalty (State cost + thrust penalty, bounded)
        thrust_penalty = thrust_action**2
        raw_step_cost = (
            self.penalty_weights[0] * current_state_cost
            + self.penalty_weights[1] * thrust_penalty
        )
        step_penalty = -np.clip(raw_step_cost, 0.0, 2.0) + mee_perturbation_penalty

        # Base Step Reward
        reward = progress_reward + step_penalty

        out_of_fuel = bool(self.propellant_mass <= 0.0)

        self.steps += 1
        truncated = bool(
            self.steps >= self.max_steps
            or self.current_time >= self.max_sim_time
            or out_of_fuel
        )
        # Termination check focused on reaching target parameter p
        terminated = bool(
            abs(self.equinoctial_parameters[0] - self.target_orbit[0])
            < self.termination_tol
        )

        # 3. Terminal Success Bonus (+10.0)
        if terminated and not out_of_fuel:
            print("Target orbit reached!")
            reward += 20.0 - mee_perturbation_penalty

        # 4. Terminal Failure Penalty (-10.0)
        elif truncated or out_of_fuel:
            if out_of_fuel:
                print("Out of fuel!")
            elif self.current_time >= self.max_sim_time:
                print("Max simulation time reached!")
            elif self.steps >= self.max_steps:
                print("Max steps reached!")
            reward *= 10.0

        info = {
            "step_times": np.array(list(state_history.keys())),
            "step_states": np.array(list(state_history.values())),
            "total_mass": self.total_mass,
            "coasting_time": coasting_time,
        }

        return (
            self.normalized_rl_state.copy(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _get_normalized_observation_and_error(self):
        mean_mass = self.initial_total_mass - (self.initial_propellant_mass / 2.0)
        half_range = self.initial_propellant_mass / 2.0
        normalized_mass = (self.total_mass - mean_mass) / half_range

        raw_errors = self.equinoctial_parameters[:5] - self.target_orbit[:5]
        scaled_errors = raw_errors / self.state_scales

        L = self.equinoctial_parameters[5]
        sin_L = np.sin(L)
        cos_L = np.cos(L)

        obs = np.concatenate((scaled_errors, [sin_L, cos_L, normalized_mass])).astype(
            np.float32
        )
        return obs, raw_errors

    def _run_propagation(self, thrust_vector, coasting_time):
        step_size_control_settings = (
            propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(
                relative_error_tolerance=self.tol,
                absolute_error_tolerance=self.tol,
            )
        )
        step_size_validation_settings = (
            propagation_setup.integrator.step_size_validation(
                minimum_step=1.0e-12, maximum_step=np.inf
            )
        )
        coefficient_set = propagation_setup.integrator.CoefficientSets.rkf_78

        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
            initial_time_step=10.0,
            coefficient_set=coefficient_set,
            step_size_control_settings=step_size_control_settings,
            step_size_validation_settings=step_size_validation_settings,
        )

        current_phase_end_time = self.current_time + coasting_time
        termination_condition = propagation_setup.propagator.time_termination(
            current_phase_end_time, terminate_exactly_on_final_condition=True
        )

        dimensional_state = self.cartesian_state.copy()
        dimensional_state[3:6] += thrust_vector
        propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            self.acceleration_models,
            self.bodies_to_propagate,
            dimensional_state,
            self.current_time,
            integrator_settings,
            termination_condition,
        )

        dynamics_simulator = simulator.create_dynamics_simulator(
            self.bodies, propagator_settings
        )
        state_history = dynamics_simulator.state_history
        del dynamics_simulator
        del propagator_settings
        return state_history
