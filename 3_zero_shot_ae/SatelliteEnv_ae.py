import numpy as np
import gymnasium as gym
import gc
import sys
import pathlib

current_dir = pathlib.Path(__file__).parent.resolve()
off_policy_dir = current_dir.parent.resolve()
project_root = off_policy_dir.parent.resolve()

for d in [str(current_dir), str(off_policy_dir), str(project_root)]:
    if d not in sys.path:
        sys.path.append(d)

from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
import tudatpy.astro.element_conversion as conversion
import tudatpy.astro.frame_conversion as frame_conversion

from utils import get_propellant_mass

from rich import print


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
        state_scales: list = [100000.0, 0.05, 0.05, 0.05, 0.05],
        termination_tol_a: float = 1e3,
        termination_tol_e: float = 1e-3,
        penalty_weights: list = [1.0, 1.0, 1.0],
        delta_a_min: float = -150000.0,
        delta_a_max: float = 150000.0,
        delta_e_min: float = -0.1,
        delta_e_max: float = 0.1,
        initial_keplerian: np.ndarray = np.empty(6, dtype=np.float64),
    ):
        super(SatelliteEnv, self).__init__()

        self.max_steps = max_steps
        self.max_sim_time = max_sim_time
        self.tol = tol
        self.initial_total_mass = initial_mass
        self.Isp = Isp
        self.initial_propellant_mass = propellant_mass
        self.initial_state = initial_state
        self.target_orbit_base = target_orbit
        self.target_orbit = target_orbit.copy()
        self.max_delta_v = max_delta_v

        self.delta_a_min = delta_a_min
        self.delta_a_max = delta_a_max
        self.delta_e_min = delta_e_min
        self.delta_e_max = delta_e_max

        self.max_coast_fraction = max_coast_fraction
        self.state_scales = np.array(state_scales, dtype=np.float32)
        self.termination_tol_a = termination_tol_a
        self.termination_tol_e = termination_tol_e

        self.penalty_weights = np.array(penalty_weights, dtype=np.float32)

        self.target_keplerian = conversion.mee_to_keplerian(self.target_orbit, True)

        # 4D Action: [Radial thrust, Along-track thrust, Cross-track thrust, Coasting time]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # 8D State: [p_err, f_err, g_err, h_err, k_err, sin(L), cos(L), mass_norm]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # ----------------------------------------------------------------------- # Tudatpy Environment Setup

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
            global_frame_origin
        ).shape_model.average_radius

        self.initial_keplerian = (
            initial_keplerian
            if initial_keplerian.size != 0
            else conversion.cartesian_to_keplerian(initial_state, self.mu)
        )

        accelerations_settings_satellite = {
            "Earth": [propagation_setup.acceleration.point_mass_gravity()]
        }
        acceleration_settings = {"Satellite": accelerations_settings_satellite}

        self.bodies_to_propagate = ["Satellite"]
        self.central_bodies = ["Earth"]
        self.acceleration_models = propagation_setup.create_acceleration_models(
            self.bodies,
            acceleration_settings,
            self.bodies_to_propagate,
            self.central_bodies,
        )

        # ----------------------------------------------------------------------- # Integrator Settings

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

        self.integrator_settings = (
            propagation_setup.integrator.runge_kutta_variable_step(
                initial_time_step=10.0,
                coefficient_set=coefficient_set,
                step_size_control_settings=step_size_control_settings,
                step_size_validation_settings=step_size_validation_settings,
            )
        )

    def reset(self, seed=None, options=None, target_delta_a=None, target_delta_e=None):
        super().reset(seed=seed)

        # ----------------------------------------------------------------------- # Reset State Variables

        self.current_time = 0.0
        self.steps = 0
        self.cartesian_state = self.initial_state.copy()

        self.total_mass = self.initial_total_mass
        self.propellant_mass = self.initial_propellant_mass

        # ----------------------------------------------------------------------- # Target Orbit Setup

        # Sample or assign zero-shot target delta_a
        if target_delta_a is not None:
            self.current_target_delta_a = float(target_delta_a)
        else:
            if not hasattr(self, "_target_sign_flip"):
                self._target_sign_flip = 1.0
            self._target_sign_flip *= -1.0  # Alternate sign on every reset
            min_abs_delta_a = 5000.0  # 5 km exclusion zone
            max_abs_delta_a = max(abs(self.delta_a_min), abs(self.delta_a_max))
            mag = float(self.np_random.uniform(min_abs_delta_a, max_abs_delta_a))
            self.current_target_delta_a = mag * self._target_sign_flip

        # Sample or assign zero-shot target delta_e
        if target_delta_e is not None:
            self.current_target_delta_e = float(target_delta_e)
        else:
            min_abs_delta_e = 0.01  # 1% exclusion zone
            delta_e = float(self.np_random.uniform(self.delta_e_min, self.delta_e_max))
            if abs(delta_e) < min_abs_delta_e:
                delta_e = min_abs_delta_e * np.sign(delta_e)  # Ensure minimum magnitude
            self.current_target_delta_e = delta_e

        # Dynamically compute target orbit MEE
        self.target_keplerian = self.initial_keplerian.copy()
        self.target_keplerian[0] += self.current_target_delta_a
        self.target_keplerian[1] += self.current_target_delta_e
        minimum_perigee_altitude = 250000.0  # 250 km above Earth's surface
        minimum_perigee_radius = self.earth_spherical_radius + minimum_perigee_altitude
        if self.target_keplerian[1] < 0.0:
            self.target_keplerian[1] = 0.0  # Ensure eccentricity is non-negative
        elif (
            self.target_keplerian[0] * (1 - self.target_keplerian[1])
            < minimum_perigee_radius
        ):
            self.target_keplerian[1] = 1 - (
                minimum_perigee_radius / self.target_keplerian[0]
            )  # Ensure perigee altitude > 250 km

        # Ensure actual target deltas match clamped target_keplerian
        self.current_target_delta_a = float(self.target_keplerian[0] - self.initial_keplerian[0])
        self.current_target_delta_e = float(self.target_keplerian[1] - self.initial_keplerian[1])

        self.target_orbit = conversion.keplerian_to_mee(self.target_keplerian)

        self.equinoctial_parameters = conversion.cartesian_to_mee(
            self.cartesian_state, self.mu
        )

        # ----------------------------------------------------------------------- # Observation and Error Computation

        self.normalized_rl_state, self.raw_errors = (
            self._get_normalized_observation_and_error()
        )

        p_init, f_init, g_init = self.equinoctial_parameters[:3]
        ecc_sq_init = f_init**2 + g_init**2
        a_init = p_init / (1.0 - ecc_sq_init) if ecc_sq_init < 1.0 else p_init
        e_init = np.sqrt(ecc_sq_init)

        init_ae_errors = np.array(
            [
                (self.target_keplerian[0] - a_init) / self.state_scales[0],
                (self.target_keplerian[1] - e_init) / self.state_scales[1],
            ],
            dtype=np.float32,
        )
        self.previous_state_cost = float(np.sum(init_ae_errors**2))

        return self.normalized_rl_state.copy(), {
            "target_delta_a": self.current_target_delta_a,
            "target_delta_e": self.current_target_delta_e,
        }

    def step(self, action):

        # ----------------------------------------------------------------------- # Action processing
        action_array = np.clip(
            np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0
        )
        thrust_action = action_array[0:3]
        coast_action = action_array[3]

        thrust_vector = thrust_action * self.max_delta_v
        thrust_vector_rsw = np.array(
            [thrust_vector[0], thrust_vector[1], thrust_vector[2]],
            dtype=np.float64,
        )

        rsw_to_inertial_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(
            self.cartesian_state
        )
        thrust_vector_inertial = rsw_to_inertial_matrix @ thrust_vector_rsw.T

        # Compute current orbital period based on equinoctial parameters
        p, f, g = self.equinoctial_parameters[:3]
        ecc_sq = f**2 + g**2
        a = p / (1.0 - ecc_sq) if ecc_sq < 1.0 else p
        current_T_orbital = 2 * np.pi * np.sqrt(max(1.0, a) ** 3 / self.mu)

        # Mapping coasting action from [-1, 1] to [min_coast_time, max_coast_time]
        min_coast_time = 1.0
        max_coast_time = self.max_coast_fraction * current_T_orbital
        coasting_time = min_coast_time + ((coast_action + 1.0) / 2.0) * (
            max_coast_time - min_coast_time
        )

        # ----------------------------------------------------------------------- # Propagation

        state_history = self._run_propagation(thrust_vector_inertial, coasting_time)

        self.cartesian_state = list(state_history.values())[-1][:6]
        self.current_time = list(state_history.keys())[-1]

        self.equinoctial_parameters = conversion.cartesian_to_mee(
            self.cartesian_state, self.mu
        )
        self.normalized_rl_state, self.raw_errors = (
            self._get_normalized_observation_and_error()
        )

        # ----------------------------------------------------------------------- # Cost computation

        # current_state_cost = np.sum(
        #     (self.raw_errors / self.state_scales) ** 2
        # )  # Cost based on normalized errors at current state

        # error_reduction = (
        #     self.previous_state_cost - current_state_cost
        # )  # Reward based on reduction in state cost w.r.t. previous state

        # self.previous_state_cost = current_state_cost

        p_new, f_new, g_new = self.equinoctial_parameters[:3]
        ecc_sq_new = f_new**2 + g_new**2
        a_current = p_new / (1.0 - ecc_sq_new) if ecc_sq_new < 1.0 else p_new
        e_current = np.sqrt(ecc_sq_new)

        ae_errors = np.array(
            [
                (self.target_keplerian[0] - a_current) / self.state_scales[0],
                (self.target_keplerian[1] - e_current) / self.state_scales[1],
            ],
            dtype=np.float32,
        )
        current_state_cost = np.sum(
            ae_errors**2
        )  # Cost based on normalized errors at current state
        error_reduction = (
            self.previous_state_cost - current_state_cost
        )  # Reward based on reduction in state cost w.r.t. previous state
        self.previous_state_cost = current_state_cost

        # ----------------------------------------------------------------------- # Reward computation

        # 1. Potential-Based Progress Reward (Unclipped for true policy invariance)
        progress_reward = 25.0 * error_reduction

        # 2. Control Effort and Error Step Penalty
        thrust_penalty = np.sum(thrust_action**2)
        step_penalty = -0.05 * np.sum(ae_errors**2) - 0.1 * thrust_penalty

        # Base Step Reward
        reward = progress_reward + step_penalty

        if (
            self.steps % 50 == 0
            or self.steps == self.max_steps - 1
            or self.current_time >= self.max_sim_time
        ):
            print(
                f"Step: {self.steps}, ae_errors: {ae_errors}, progress_reward: {progress_reward:.4f}, step_penalty: {step_penalty:.4f}, total_reward: {reward:.4f}"
            )

        # ----------------------------------------------------------------------- # Mass Update and Truncation Check

        self.propellant_mass -= get_propellant_mass(
            np.linalg.norm(thrust_vector_inertial), self.Isp, self.total_mass
        )
        self.total_mass = self.initial_total_mass - (
            self.initial_propellant_mass - self.propellant_mass
        )
        out_of_fuel = bool(self.propellant_mass <= 0.0)

        truncated = bool(
            self.steps >= self.max_steps - 1
            or self.current_time >= self.max_sim_time
            or out_of_fuel
        )

        # ----------------------------------------------------------------------- # Termination Check

        # Termination check
        a_reached = bool(
            abs(ae_errors[0] * self.state_scales[0]) < self.termination_tol_a
        )
        e_reached = bool(
            abs(ae_errors[1] * self.state_scales[1]) < self.termination_tol_e
        )
        terminated = bool(a_reached and e_reached)

        # ----------------------------------------------------------------------- # Terminal Reward Adjustments

        # Terminal Success Bonus (+50.0)
        if terminated and not out_of_fuel:
            print("[green]Target orbit reached![/green]")
            reward += 50.0

        # ----------------------------------------------------------------------- # Info Dictionary and Return

        info = {
            "step_times": np.array(list(state_history.keys())),
            "step_states": np.array(list(state_history.values())),
            "total_mass": self.total_mass,
            "coasting_time": coasting_time,
            "target_delta_a": self.current_target_delta_a,
            "target_delta_e": self.current_target_delta_e,
            "ae_errors": ae_errors,
        }
        # print(
        #     f"Step: {self.steps}, Time: {self.current_time:.2f}s, Reward: {reward:.4f}, a_err: {ae_errors[0]*self.state_scales[0]:.2f} m, e_err: {ae_errors[1]*self.state_scales[1]:.4f}, Mass: {self.total_mass:.2f} kg"
        # )
        self.steps += 1
        if truncated:
            print(
                f"[red]Episode truncated due to: {'out of fuel' if out_of_fuel else 'max steps' if self.steps >= self.max_steps - 1 else 'max simulation time'}![/red]"
            )

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

        raw_errors = self.target_orbit[:5] - self.equinoctial_parameters[:5]
        scaled_errors = raw_errors / self.state_scales

        L = self.equinoctial_parameters[5]
        sin_L = np.sin(L)
        cos_L = np.cos(L)

        obs = np.concatenate((scaled_errors, [sin_L, cos_L, normalized_mass])).astype(
            np.float32
        )
        return obs, raw_errors

    def _run_propagation(self, thrust_vector, coasting_time):

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
            self.integrator_settings,
            termination_condition,
        )

        dynamics_simulator = simulator.create_dynamics_simulator(
            self.bodies, propagator_settings
        )
        state_history = dynamics_simulator.state_history
        del dynamics_simulator
        del propagator_settings

        return state_history
