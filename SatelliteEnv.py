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
    def __init__(self, max_steps: int, max_sim_time: float, tol: float, initial_mass: float, propellant_mass: float, Isp: float, initial_state: np.ndarray, target_coordinates: np.ndarray, max_delta_v: float, max_coast_time: float):
        super(SatelliteEnv, self).__init__()

        

        ############################################################################
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


        ###########################################################################
        ### Action and observation space
        # Actions are continuous thrust values in 2D space (Along-track and Cross-track) + Flight time after burn
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Observation space includes position and velocity in 3D space, objective coordinates (lat, lon), remaining propellant mass 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        ###########################################################################
        # Space environment
        spice.load_standard_kernels()  # Load spice kernels.
        bodies_to_create = ['Earth']
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
        body_settings.get( "Earth" ).shape_settings = environment_setup.shape.spherical_spice( )
        # self.earth_spherical_radius = body_settings.get( "Earth" ).shape_settings.SphericalBodyShapeSettings.radius
        self.earth_spherical_radius = self.bodies.get("Earth").shape_model.average_radius
        


        ###########################################################################
        # CREATE ACCELERATIONS 
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
            self.bodies, acceleration_settings, self.bodies_to_propagate, self.central_bodies
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #########################################################################
        # Reset time
        self.current_time = 0.0
        self.steps = 0
        # #########################################################################
        # # Keplerian elements for initial orbit around Earth
        # altitude = 500000.0  # 500 km altitude
        # eccentricity = 0.01  # Small eccentricity
        # inclination = np.radians(45.0)  # 45 degrees inclination
        # earth_radius = self.bodies.get("Earth").shape_model.average_radius
        # self.r_apogee = earth_radius + altitude * (1 + eccentricity)
        # self.r_perigee = earth_radius + altitude * (1 - eccentricity)
        # semi_major_axis = (self.r_apogee + self.r_perigee) / 2
        # RAAN = np.radians(0.0)  # Right Ascension of Ascending Node
        # argument_of_periapsis = np.radians(0.0)  # Argument of Periapsis
        # true_anomaly = np.radians(0.0)

        # semilatus_rectum = semi_major_axis * (1 - eccentricity**2)
        # central_body_gravitational_parameter = self.bodies.get( self.central_bodies[0] ).gravitational_parameter

        # self.v_perigee = np.sqrt(central_body_gravitational_parameter  / semilatus_rectum) * (1 + eccentricity)
        # self.v_apogee = np.sqrt(central_body_gravitational_parameter  / semilatus_rectum) * (1 - eccentricity)
        
        # ### Cartesian state from Keplerian elements
        # self.cartesian_state = conversion.keplerian_to_cartesian(
        #     semi_major_axis,
        #     eccentricity,
        #     inclination,
        #     argument_of_periapsis,
        #     RAAN,
        #     true_anomaly,
        #     central_body_gravitational_parameter
        # )

        self.cartesian_state = self.initial_state.copy()

        # #########################################################################
        # ### Target state - random latitude and longitude on Earth
        # target_latitude = np.radians(np.random.uniform(-90, 90))
        # target_longitude = np.radians(np.random.uniform(-180, 180))

        # self.target_point = np.array([target_latitude, target_longitude])
        self.target_point = self.target_coordinates.copy()

        # #########################################################################
        # ### Satellite mass 
        # self.mass = self.bodies.get('Satellite').mass = 5e2 
        # self.propellant_mass = 1e2  # 100 kg of propellant

        self.total_mass = self.initial_total_mass
        self.propellant_mass = self.initial_propellant_mass
        self.rl_state = np.concatenate((self.cartesian_state, self.target_point, [self.propellant_mass]))
        self.normalized_state = self._get_normalized_observation()
        
        return self.normalized_state.copy(), {}
    

    def step(self, action):
        # Update the state based on the action (thrust)
        # Action is a 2D Delta V vector + time after burn
        thrust_vector_rsw = np.concatenate(([0.0], action[:2] * self.max_delta_v)) # Extract the rsw thrust vector [km/s]
        rsw_to_inertial_matrix = frame_conversion.rsw_to_inertial_rotation_matrix(self.cartesian_state)  # Get the RSW to inertial transformation matrix
        thrust_vector_intertial = rsw_to_inertial_matrix @ thrust_vector_rsw.T  # Transform thrust vector to inertial frame
        coasting_time = self.max_coast_time * (action[2] + 1) / 2  # Extract the coasting time [s]

        # Propellant mass consumption based on the thrust applied, and update the total mass of the spacecraft
        self.propellant_mass -= get_propellant_mass(np.linalg.norm(thrust_vector_intertial), self.Isp, self.total_mass)  # Update propellant mass based on thrust
        self.total_mass = self.initial_total_mass - (self.initial_propellant_mass - self.propellant_mass)  # Update total mass based on propellant used

        # Running dynamic simulation for given thrust vector and coasting time 
        state_history = self._run_propagation(thrust_vector_intertial, coasting_time)

        # Update of state after propagation
        self.cartesian_state =  list(state_history.values())[-1][:6]  # Update the cartesian state after propagation
        self.current_time =  list(state_history.keys())[-1]  # Update current time to the end of propagation
        self.rl_state = np.concatenate((self.cartesian_state, self.target_point, [self.propellant_mass]))
        self.normalized_state = self._get_normalized_observation()  # Update the state after propagation

        # Computation of spherical distance to target
        self.spherical_distance = compute_spherical_ground_distance(self.cartesian_state, self.target_point, self.earth_spherical_radius)  # Compute the current coordinates of the satellite
        
        # Calculate reward (negative distance to target)
        reward = -(self.spherical_distance / (np.pi * self.earth_spherical_radius) + np.linalg.norm(thrust_vector_intertial) / self.max_delta_v )  # Reward is negative distance to target, normalized by Earth's circumference, plus penalty for thrust usage

        # Check if the episode is done (if close enough to target)
        terminated = self.spherical_distance < 0.1 # 100 meters threshold for reaching the target

        # Truncation check
        self.steps += 1
        truncated = self.steps >= self.max_steps or self.current_time >= self.max_sim_time or self.propellant_mass <= 0.0

        return self.normalized_state.copy(), reward, terminated, truncated, {}

    def render(self):
        pass  # Rendering logic can be implemented here if needed


    def _get_normalized_observation(self):
        # Normalize the observation to be in the range [-1, 1]
        self.keplerian_elements = conversion.cartesian_to_keplerian(self.cartesian_state, self.mu)
        semi_major_axis = self.keplerian_elements[0]
        self.eccentricity = self.keplerian_elements[1]
        self.r_apogee = semi_major_axis * (1 + self.eccentricity)
        semilatus_rectum = semi_major_axis * (1 - self.eccentricity**2)
        self.v_perigee = np.sqrt(self.mu / semilatus_rectum) * (1 + self.eccentricity)

        mean_mass = self.initial_total_mass - (self.initial_propellant_mass / 2)
        half_range = self.initial_propellant_mass / 2

        normalized_position = self.cartesian_state[:3] / self.r_apogee
        normalized_velocity = self.cartesian_state[3:6] / self.v_perigee
        normalized_target = self.target_coordinates[:2] / np.array([np.pi/2, np.pi])  # Normalize lat/lon
        normalized_mass = (self.rl_state[8] - mean_mass) / half_range  # Normalize mass

        return np.concatenate((normalized_position, normalized_velocity, normalized_target, [normalized_mass]))
    

    def _get_dimensional_observation(self, normalized_observation):
        # Convert normalized observation back to dimensional values
        position = normalized_observation[:3] * self.r_apogee
        velocity = normalized_observation[3:6] * self.v_perigee
        target_latitude = normalized_observation[6] * (np.pi/2)
        target_longitude = normalized_observation[7] * np.pi
        mass = normalized_observation[8] * (self.propellant_mass / 2) + (self.mass - (self.propellant_mass / 2))

        return np.concatenate((position, velocity, [target_latitude, target_longitude], [mass]))
    

    def _run_propagation(self, thrust_vector, coasting_time):
        """
        Run the propagation for the given thrust vector and coasting time.
        :param thrust_vector: The thrust vector to apply (2D) [km/s].
        :param coasting_time: The time to propagate after applying the thrust [s].
        """
        # Implement the propagation logic using TUDAT here
        # This function should update self.state based on the thrust_vector and coasting_time

        ###########################################################################
        # Define the integrator settings
        step_size_control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(
            relative_error_tolerance=self.tol,
            absolute_error_tolerance=self.tol,
        )
        step_size_validation_settings = propagation_setup.integrator.step_size_validation(
            minimum_step=1.0e-12,
            maximum_step=np.inf
        )
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78

        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
            initial_time_step=250.0,
            coefficient_set=coefficient_set,
            step_size_control_settings=step_size_control_settings,
            step_size_validation_settings=step_size_validation_settings 
        )

        # Define termination conditions (enforce exact termination time)
        current_phase_end_time = self.current_time + coasting_time
        termination_condition = propagation_setup.propagator.time_termination(
            current_phase_end_time, terminate_exactly_on_final_condition=True
        )


        # dimensional_state = self._get_dimensional_observation(self.state)[:6]  # Current state (position and velocity)
        dimensional_state = self.cartesian_state.copy()  # Current state (position and velocity)
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


        dynamics_simulator = simulator.create_dynamics_simulator(self.bodies, propagator_settings)
        return dynamics_simulator.state_history


if __name__ == "__main__":
    
    # Parametri fittizi per inizializzare l'ambiente
    target_coordinates = np.array([np.radians(45.0), np.radians(45.0)]) # lat, lon


    #########################################################################
    # Keplerian elements for initial orbit around Earth
    altitude = 500000.0  # 500 km altitude
    eccentricity = 0.01  # Small eccentricity
    inclination = np.radians(45.0)  # 45 degrees inclination
    earth_radius = 6378
    r_apogee = earth_radius + altitude * (1 + eccentricity)
    r_perigee = earth_radius + altitude * (1 - eccentricity)
    semi_major_axis = (r_apogee + r_perigee) / 2
    RAAN = np.radians(0.0)  # Right Ascension of Ascending Node
    argument_of_periapsis = np.radians(0.0)  # Argument of Periapsis
    true_anomaly = np.radians(0.0)

    semilatus_rectum = semi_major_axis * (1 - eccentricity**2)
    central_body_gravitational_parameter = 398600.4418  # Earth's gravitational parameter in km^3/s^2

    v_perigee = np.sqrt(central_body_gravitational_parameter  / semilatus_rectum) * (1 + eccentricity)
    v_apogee = np.sqrt(central_body_gravitational_parameter  / semilatus_rectum) * (1 - eccentricity)
        
    ### Cartesian state from Keplerian elements
    cartesian_state = conversion.keplerian_to_cartesian(np.array([
            semi_major_axis,
            eccentricity,
            inclination,
            argument_of_periapsis,
            RAAN,
            true_anomaly,
    ]),
            central_body_gravitational_parameter

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
        max_delta_v=0.1, # 100 m/s max per step
        max_coast_time=3600.0 # 1 hour max per step
    )

    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")

    print("\nStarting test with random actions...")
    for i in range(50):
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
 
