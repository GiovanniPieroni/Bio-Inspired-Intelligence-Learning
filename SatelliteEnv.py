import numpy as np

# RL environment imports
import gymnasium as gym

# TUDAT imports
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
import tudatpy.astro.element_conversion as conversion
import tudatpy.astro.frame_conversion as frame_conversion

# Utils import
from utils import get_propellant_mass, compute_spherical_ground_distance


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
        target_coordinates: np.ndarray,
        max_delta_v: float,
        max_coast_time: float,
    ):
        super(SatelliteEnv, self).__init__()

        """
        Initialize the satellite environment for reinforcement learning.
        """

        ### Assignment of parameters
        self.max_steps = max_steps
        self.max_sim_time = max_sim_time
        self.tol = tol
        self.initial_total_mass = initial_mass
        self.Isp = Isp
        self.initial_propellant_mass = propellant_mass
        self.initial_state = initial_state
        self.target_coordinates = target_coordinates
        self.max_delta_v = max_delta_v
        self.max_coast_time = max_coast_time

        ### Action and observation space
        # Actions are continuous thrust values in 2D space (Along-track and Cross-track) + Flight time after burn
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        # Observation space includes position and velocity in 3D space, objective coordinates in ECEF (lat, lon), remaining propellant mass
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Space environment
        spice.load_standard_kernels()  # Load spice kernels.
        bodies_to_create = ["Earth"]
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation
        )

        # Add satellite body to the environment
        body_settings.add_empty_settings("Satellite")

        # Create environment and parameters for Earth
        self.bodies = environment_setup.create_system_of_bodies(body_settings)
        self.mu = self.bodies.get(global_frame_origin).gravitational_parameter

        # create spherical shape model settings
        body_settings.get("Earth").shape_settings = (
            environment_setup.shape.spherical_spice()
        )
        # self.earth_spherical_radius = body_settings.get( "Earth" ).shape_settings.SphericalBodyShapeSettings.radius
        self.earth_spherical_radius = self.bodies.get(
            "Earth"
        ).shape_model.average_radius

        self.omega_earth = 2 * np.pi / 86164.0905

        lat_t, lon_t = self.target_coordinates
        self.target_ECEF = np.array(
            [
                self.earth_spherical_radius * np.cos(lat_t) * np.cos(lon_t),
                self.earth_spherical_radius * np.cos(lat_t) * np.sin(lon_t),
                self.earth_spherical_radius * np.sin(lat_t),
            ]
        )

        ### CREATE ACCELERATIONS
        # Define bodies that are propagated, and their central bodies of propagation.
        self.bodies_to_propagate = ["Satellite"]
        self.central_bodies = ["Earth"]

        # Define accelerations acting on vehicle.
        acceleration_settings_on_vehicle = dict(
            Earth=[propagation_setup.acceleration.point_mass_gravity()]
        )

        # Create global accelerations dictionary.
        acceleration_settings = {"Satellite": acceleration_settings_on_vehicle}

        # Create acceleration models.
        self.acceleration_models = propagation_setup.create_acceleration_models(
            self.bodies,
            acceleration_settings,
            self.bodies_to_propagate,
            self.central_bodies,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset time
        self.current_time = 0.0
        self.steps = 0

        # Initial state in J2000 coordinates
        self.cartesian_state = self.initial_state.copy()

        # Initial mass
        self.total_mass = self.initial_total_mass
        self.propellant_mass = self.initial_propellant_mass

        # At time t=0, the J2000 and ECEF frames are aligned, so the conversion is direct.
        sat_pos_ECEF = self.cartesian_state[:3]

        # Build the new 10-dimensional RL state (Position, Velocity, Target_ECEF, Mass)
        self.rl_state = np.concatenate(
            (
                sat_pos_ECEF,
                self.cartesian_state[3:6],
                self.target_ECEF,
                [self.propellant_mass],
            )
        )
        self.normalized_state = self._get_normalized_observation()

        norm_sat = np.linalg.norm(sat_pos_ECEF)
        norm_target = np.linalg.norm(self.target_ECEF)

        # Dot product to compute the angle gamma, clipped for numerical stability
        cos_gamma = np.clip(
            np.dot(sat_pos_ECEF, self.target_ECEF) / (norm_sat * norm_target), -1.0, 1.0
        )
        gamma = np.arccos(cos_gamma)

        # Surface distance in meters at time 0
        self.spherical_distance = self.earth_spherical_radius * gamma

        return self.normalized_state.copy(), {}

    def step(self, action):
        # Update the state based on the action (thrust)
        # Action is a 2D Delta V vector + time after burn
        thrust_vector_rsw = np.concatenate(
            ([0.0], action[:2] * self.max_delta_v)
        )  # Extract the rsw thrust vector [km/s]

        rsw_to_inertial_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(
            self.cartesian_state
        )  # Get the RSW to inertial transformation matrix

        thrust_vector_inertial = (
            rsw_to_inertial_matrix @ thrust_vector_rsw.T
        )  # Transform thrust vector to inertial frame

        coasting_time = (
            self.max_coast_time * (action[2] + 1) / 2
        )  # Extract the coasting time [s]

        # Propellant mass consumption based on the thrust applied
        self.propellant_mass -= get_propellant_mass(
            np.linalg.norm(thrust_vector_inertial), self.Isp, self.total_mass
        )
        self.total_mass = self.initial_total_mass - (
            self.initial_propellant_mass - self.propellant_mass
        )

        # Running dynamic simulation for given thrust vector and coasting time
        state_history = self._run_propagation(thrust_vector_inertial, coasting_time)

        # Coasting times and states from the propagation history, for potential analysis or plotting purposes
        coast_times = np.array(list(state_history.keys()))
        coast_states = np.array(list(state_history.values()))

        info = {"coast_times": coast_times, "coast_states": coast_states}

        # Update of state after propagation (J2000 Inertial)
        self.cartesian_state = list(state_history.values())[-1][:6]
        self.current_time = list(state_history.keys())[-1]

        # --- ECEF TRANSFORMATION ---
        # 1. Rotation matrix R_z
        theta = self.omega_earth * self.current_time
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_z = np.array([[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])

        sat_pos_J2000 = self.cartesian_state[:3]
        sat_vel_J2000 = self.cartesian_state[3:6]

        # 2. Position in ECEF
        sat_pos_ECEF = R_z @ sat_pos_J2000

        # 3. Velocity in ECEF (including Coriolis/drift term)
        cross_omega_r = np.array(
            [
                -self.omega_earth * sat_pos_ECEF[1],
                self.omega_earth * sat_pos_ECEF[0],
                0.0,
            ]
        )
        sat_vel_ECEF = (R_z @ sat_vel_J2000) - cross_omega_r

        # --- ANGULAR DISTANCE AND GROUND TRACK CALCULATION ---
        norm_sat = np.linalg.norm(sat_pos_ECEF)
        norm_target = np.linalg.norm(self.target_ECEF)

        # Dot product to compute the angle gamma, clipped for numerical stability
        cos_gamma = np.clip(
            np.dot(sat_pos_ECEF, self.target_ECEF) / (norm_sat * norm_target), -1.0, 1.0
        )
        gamma = np.arccos(cos_gamma)

        # Surface distance in meters
        self.spherical_distance = self.earth_spherical_radius * gamma

        # --- RL STATE UPDATE ---
        # The state now includes the target's 3D ECEF coordinates (total dimension: 10)
        self.rl_state = np.concatenate(
            (sat_pos_ECEF, sat_vel_ECEF, self.target_ECEF, [self.propellant_mass])
        )
        self.normalized_state = self._get_normalized_observation()

        # --- REWARD AND TERMINATION CONDITIONS ---
        reward = -(  # Reward is negative distance to target, normalized by Earth's circumference, plus penalty for thrust usage
            self.spherical_distance / (np.pi * self.earth_spherical_radius)
            + np.linalg.norm(thrust_vector_inertial) / self.max_delta_v
        )

        # Check if the episode is done (e.g. threshold of 10 km from the target)
        terminated = bool(self.spherical_distance < 10000.0)

        self.steps += 1
        truncated = bool(
            self.steps >= self.max_steps
            or self.current_time >= self.max_sim_time
            or self.propellant_mass <= 0.0
        )

        return self.normalized_state.copy(), float(reward), terminated, truncated, info

    def _get_normalized_observation(self):
        # Normalize the observation to be in the range [-1, 1]
        self.keplerian_elements = conversion.cartesian_to_keplerian(
            self.cartesian_state, self.mu
        )
        semi_major_axis = self.keplerian_elements[0]
        self.eccentricity = self.keplerian_elements[1]
        self.r_apogee = semi_major_axis * (1 + self.eccentricity)
        semilatus_rectum = semi_major_axis * (1 - self.eccentricity**2)
        self.v_perigee = np.sqrt(self.mu / semilatus_rectum) * (1 + self.eccentricity)

        mean_mass = self.initial_total_mass - (self.initial_propellant_mass / 2)
        half_range = self.initial_propellant_mass / 2

        # Correct extraction from the new 10-dimensional rl_state
        normalized_position = self.rl_state[:3] / self.r_apogee
        normalized_velocity = self.rl_state[3:6] / self.v_perigee
        normalized_target = self.rl_state[6:9] / self.earth_spherical_radius
        normalized_mass = (
            self.rl_state[9] - mean_mass  # Now mass is element 9
        ) / half_range  # Ora la massa è l'elemento 9

        return np.concatenate(
            (
                normalized_position,
                normalized_velocity,
                normalized_target,
                [normalized_mass],
            )
        )

    def _get_dimensional_observation(self, normalized_observation):
        # Convert normalized observation back to dimensional values
        position = normalized_observation[:3] * self.r_apogee
        velocity = normalized_observation[3:6] * self.v_perigee
        target_latitude = normalized_observation[6] * (np.pi / 2)
        target_longitude = normalized_observation[7] * np.pi
        mass = normalized_observation[8] * (self.propellant_mass / 2) + (
            self.mass - (self.propellant_mass / 2)
        )

        return np.concatenate(
            (position, velocity, [target_latitude, target_longitude], [mass])
        )

    def _run_propagation(self, thrust_vector, coasting_time):
        """
        Run the propagation for the given thrust vector and coasting time.
        :param thrust_vector: The thrust vector to apply (2D) [km/s].
        :param coasting_time: The time to propagate after applying the thrust [s].
        """

        ###########################################################################
        # Define the integrator settings
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

        # Define termination conditions (enforce exact termination time)
        current_phase_end_time = self.current_time + coasting_time
        termination_condition = propagation_setup.propagator.time_termination(
            current_phase_end_time, terminate_exactly_on_final_condition=True
        )

        # dimensional_state = self._get_dimensional_observation(self.state)[:6]  # Current state (position and velocity)
        dimensional_state = (
            self.cartesian_state.copy()
        )  # Current state (position and velocity)
        dimensional_state[3:6] += thrust_vector  # Apply thrust to velocity
        propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            self.acceleration_models,
            self.bodies_to_propagate,
            dimensional_state,  # Current state (position and velocity + Delta V)
            self.current_time,
            integrator_settings,
            termination_condition,
        )

        dynamics_simulator = simulator.create_dynamics_simulator(
            self.bodies, propagator_settings
        )
        return dynamics_simulator.state_history


if __name__ == "__main__":

    # Define target coordinates (latitude, longitude) in radians
    target_coordinates = np.array([np.radians(45.0), np.radians(45.0)])  # lat, lon

    #########################################################################
    # Keplerian elements for initial orbit around Earth
    altitude = 500000.0  # 500 km altitude
    eccentricity = 0.01  # Small eccentricity
    inclination = np.radians(45.0)  # 45 degrees inclination
    earth_radius = 6378000
    r_apogee = earth_radius + altitude * (1 + eccentricity)
    r_perigee = earth_radius + altitude * (1 - eccentricity)
    semi_major_axis = (r_apogee + r_perigee) / 2
    RAAN = np.radians(0.0)  # Right Ascension of Ascending Node
    argument_of_periapsis = np.radians(0.0)  # Argument of Periapsis
    true_anomaly = np.radians(0.0)

    semilatus_rectum = semi_major_axis * (1 - eccentricity**2)
    central_body_gravitational_parameter = (
        398600.4418e9  # Earth's gravitational parameter in m^3/s^2
    )

    v_perigee = np.sqrt(central_body_gravitational_parameter / semilatus_rectum) * (
        1 + eccentricity
    )
    v_apogee = np.sqrt(central_body_gravitational_parameter / semilatus_rectum) * (
        1 - eccentricity
    )

    ### Cartesian state from Keplerian elements
    cartesian_state = conversion.keplerian_to_cartesian(
        np.array(
            [
                semi_major_axis,
                eccentricity,
                inclination,
                argument_of_periapsis,
                RAAN,
                true_anomaly,
            ]
        ),
        central_body_gravitational_parameter,
    )

    env = SatelliteEnv(
        max_steps=100,
        max_sim_time=86400.0,
        tol=1e-8,
        initial_mass=500.0,
        propellant_mass=100.0,
        Isp=300.0,
        initial_state=cartesian_state,
        target_coordinates=target_coordinates,
        max_delta_v=0.1,  # 100 m/s max per step
        max_coast_time=3600.0,  # 1 hour max per step
    )

    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")

    print("\nStarting test with random actions...")
    for i in range(5):
        # The agent chooses a completely random action from the defined space
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {i+1}:")
        print(f"  Action taken: {action}")
        print(f"  Reward obtained: {reward:.4f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}\n")

        if terminated or truncated:
            print("Episode completed prematurely.")
            break

    print("Test completed.")
