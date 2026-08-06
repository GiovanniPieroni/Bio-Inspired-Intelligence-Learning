from tabnanny import verbose

import numpy as np

from tudatpy.astro import element_conversion as conversion


def get_propellant_mass(delta_v, isp, initial_mass):
    """
    Calculate the propellant mass required for a given delta-v using the Tsiolkovsky rocket equation.

    Parameters:
    delta_v (float): The required change in velocity (m/s).
    isp (float): The specific impulse of the engine (s).
    initial_mass (float): The initial mass of the spacecraft (kg).

    Returns:
    float: The required propellant mass (kg).
    """
    g0 = 9.80665  # Standard gravity in m/s^2
    propellant_mass = initial_mass * (1 - np.exp(-delta_v / (isp * g0)))
    return propellant_mass


def compute_spherical_ground_distance(state, target_coordinates, earth_radius):
    """
    Compute the ground distance between the current state and the target coordinates.

    Parameters:
    state (np.ndarray): The current state of the spacecraft (position and velocity).
    target_coordinates (np.ndarray): The target coordinates (latitude, longitude).

    Returns:
    float: The ground distance to the target (m).
    """
    # Extract position from state
    position = state[:3]  # State is [x, y, z, vx, vy, vz]

    # Convert position to latitude and longitude
    r = np.linalg.norm(position)
    lat = np.arcsin(position[2] / r)  # Latitude in radians
    lon = np.arctan2(position[1], position[0])  # Longitude in radians

    target_lat = target_coordinates[0]
    target_lon = target_coordinates[1]

    # Haversine formula to calculate distance
    dlat = target_lat - lat
    dlon = target_lon - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(target_lat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius in meters
    R = earth_radius
    distance = R * c

    return distance


def generate_random_target_orbit(
    deltav_max, initial_orbit, earth_radius=6378000.0, mu=398600.4418e9, verbose=False
):
    """
    Generates a random target orbit in LEO with physically reachable perturbations from the nominal orbit. The limit is given from the maximum delta-v given.
    The function returns the Modified Equinoctial Elements (MEE) of the target orbit.
    :param deltav_max: Maximum delta-v allowed for the perturbation (m/s)
    :param initial_orbit: The initial orbit in Keplerian format (numpy array)
    :param mu: Gravitational parameter of the central body (default is Earth's mu)
    """
    # Compute the initial Cartesian state from the given Keplerian elements
    initial_cartesian_state = conversion.keplerian_to_cartesian(initial_orbit, mu)
    v_initial = np.linalg.norm(
        initial_cartesian_state[3:]
    )  # Initial velocity magnitude

    a_i = initial_orbit[0]
    e_i = initial_orbit[1]
    i_i = initial_orbit[2]

    # Calculate the maximum change in semi-major axis based on the delta-v limit
    delta_a_max = (a_i**2 / mu * (deltav_max**2 + 2 * deltav_max * v_initial)) / (
        1 - a_i / mu * (deltav_max**2 + 2 * deltav_max * v_initial)
    )

    a_min = max(earth_radius + 200000.0, a_i - delta_a_max)
    a_max = a_i + delta_a_max

    r_p_min = (
        200000.0 + earth_radius
    )  # Minimum perigee radius (200 km above Earth's surface)

    # Calculate the maximum eccentricity based on the semi-major axis limit
    e_max = 1 - r_p_min / a_max

    e_min = 0.0  # Minimum eccentricity (circular orbit)

    # Generate random semi-major axis and eccentricity within the calculated limits
    a_target = np.random.uniform(a_min, a_max)
    e_target = np.random.uniform(e_min, e_max)

    # Calculate the maximum inclination change based on the delta-v limit
    delta_i_max = 2 * np.arcsin(deltav_max / (v_initial * 2))

    i_target = np.random.uniform(
        max(0.0, initial_orbit[2] - delta_i_max),
        min(np.pi, initial_orbit[2] + delta_i_max),
    )

    # Calculate the maximum change in argument of perigee and RAAN based on the delta-v limit
    v_r_max = np.sqrt(mu * e_i**2 / (a_i * (1 - e_i**2)))
    if deltav_max > v_r_max * 2:
        delta_omega_max = np.pi  # Allow full range if delta-v is sufficient
    else:
        delta_omega_max = 2 * np.arcsin(deltav_max / (v_r_max * 2))

    v_i_min = np.sqrt(mu * (1 + e_i) / (a_i * (1 - e_i)))
    delta_raan_max = 2 * np.arcsin(deltav_max / (v_i_min * 2 * np.sin(i_i)))

    omega_target = np.random.uniform(
        initial_orbit[3] - delta_omega_max, initial_orbit[3] + delta_omega_max
    )
    raan_target = np.random.uniform(
        initial_orbit[4] - delta_raan_max, initial_orbit[4] + delta_raan_max
    )

    # Keep 0 as the true anomaly for the target orbit
    theta_target = 0.0

    target_keplerian = np.array(
        [a_target, e_target, i_target, omega_target, raan_target, theta_target]
    )

    target_mee = conversion.keplerian_to_mee(target_keplerian)
    # verbose = True
    if verbose:
        print("Target Keplerian Elements:")
        print(f"Semi-major axis (a): {a_target:.2f} m")
        print(f"Eccentricity (e): {e_target:.4f}")
        print(f"Inclination (i): {np.degrees(i_target):.2f} degrees")
        print(f"Argument of Perigee (ω): {np.degrees(omega_target):.2f} degrees")
        print(f"RAAN (Ω): {np.degrees(raan_target):.2f} degrees")
        print(f"True Anomaly (θ): {np.degrees(theta_target):.2f} degrees")

    return target_mee


def generate_random_target_orbit_single(
    deltav_max,
    initial_orbit,
    parameters=None,
    earth_radius=6378000.0,
    mu=398600.4418e9,
    verbose=False,
):
    """
    Generates a random target orbit in LEO with physically reachable perturbations from the nominal orbit. The limit is given from the maximum delta-v given.
    The function returns the Modified Equinoctial Elements (MEE) of the target orbit.
    This function can be used to modify all or only some of the orbital elements, depending on the requirements.
    :param deltav_max: Maximum delta-v allowed for the perturbation (m/s)
    :param initial_orbit: The initial orbit in Keplerian format (numpy array)
    :param parameters: List of orbital elements to modify (default is all elements)
    :param mu: Gravitational parameter of the central body (default is Earth's mu)
    """
    # Compute the initial Cartesian state from the given Keplerian elements
    initial_cartesian_state = conversion.keplerian_to_cartesian(initial_orbit, mu)
    v_initial = np.linalg.norm(
        initial_cartesian_state[3:]
    )  # Initial velocity magnitude

    a_i = initial_orbit[0]
    e_i = initial_orbit[1]
    i_i = initial_orbit[2]

    # Calculate the maximum change in semi-major axis based on the delta-v limit
    delta_a_max = (a_i**2 / mu * (deltav_max**2 + 2 * deltav_max * v_initial)) / (
        1 - a_i / mu * (deltav_max**2 + 2 * deltav_max * v_initial)
    )

    a_min = max(earth_radius + 200000.0, a_i - delta_a_max)
    a_max = a_i + delta_a_max

    r_p_min = (
        200000.0 + earth_radius
    )  # Minimum perigee radius (200 km above Earth's surface)

    # Calculate the maximum eccentricity based on the semi-major axis limit
    e_max = 1 - r_p_min / a_max

    e_min = 0.0  # Minimum eccentricity (circular orbit)

    # Generate random semi-major axis and eccentricity within the calculated limits
    a_target = np.random.uniform(a_min, a_max)
    e_target = np.random.uniform(e_min, e_max)

    # Calculate the maximum inclination change based on the delta-v limit
    delta_i_max = 2 * np.arcsin(deltav_max / (v_initial * 2))

    i_target = np.random.uniform(
        max(0.0, initial_orbit[2] - delta_i_max),
        min(np.pi, initial_orbit[2] + delta_i_max),
    )

    # Calculate the maximum change in argument of perigee and RAAN based on the delta-v limit
    v_r_max = np.sqrt(mu * e_i**2 / (a_i * (1 - e_i**2)))
    if deltav_max > v_r_max * 2:
        delta_omega_max = np.pi  # Allow full range if delta-v is sufficient
    else:
        delta_omega_max = 2 * np.arcsin(deltav_max / (v_r_max * 2))

    v_i_min = np.sqrt(mu * (1 + e_i) / (a_i * (1 - e_i)))
    delta_raan_max = 2 * np.arcsin(deltav_max / (v_i_min * 2 * np.sin(i_i)))

    omega_target = np.random.uniform(
        initial_orbit[3] - delta_omega_max, initial_orbit[3] + delta_omega_max
    )
    raan_target = np.random.uniform(
        initial_orbit[4] - delta_raan_max, initial_orbit[4] + delta_raan_max
    )

    # Keep 0 as the true anomaly for the target orbit
    theta_target = 0.0

    target_keplerian = np.array(
        [a_target, e_target, i_target, omega_target, raan_target, theta_target]
    )

    target_mee = conversion.keplerian_to_mee(target_keplerian)

    if parameters is not None:
        if "p" not in parameters:
            target_mee[0] = conversion.keplerian_to_mee(initial_orbit)[0]
        if "f" not in parameters:
            target_mee[1] = conversion.keplerian_to_mee(initial_orbit)[1]
        if "g" not in parameters:
            target_mee[2] = conversion.keplerian_to_mee(initial_orbit)[2]
        if "h" not in parameters:
            target_mee[3] = conversion.keplerian_to_mee(initial_orbit)[3]
        if "k" not in parameters:
            target_mee[4] = conversion.keplerian_to_mee(initial_orbit)[4]
        if "L" not in parameters:
            target_mee[5] = conversion.keplerian_to_mee(initial_orbit)[5]

    # verbose = True
    if verbose:
        print("Target Keplerian Elements:")
        print(f"Semi-major axis (a): {a_target:.2f} m")
        print(f"Eccentricity (e): {e_target:.4f}")
        print(f"Inclination (i): {np.degrees(i_target):.2f} degrees")
        print(f"Argument of Perigee (ω): {np.degrees(omega_target):.2f} degrees")
        print(f"RAAN (Ω): {np.degrees(raan_target):.2f} degrees")
        print(f"True Anomaly (θ): {np.degrees(theta_target):.2f} degrees")

    return target_mee


if __name__ == "__main__":
    from config import load_config

    # Load configuration from config.yaml
    cfg = load_config()

    # Construct default initial Keplerian orbit
    initial_keplerian_orbit = np.array(
        [
            cfg.orbit.semi_major_axis,
            cfg.orbit.eccentricity,
            cfg.orbit.inclination_rad,
            cfg.orbit.arg_of_periapsis_rad,
            cfg.orbit.raan_rad,
            cfg.orbit.true_anomaly_rad,
        ]
    )

    # Convert initial orbit to modified equinoctial elements (MEE)
    initial_mee = conversion.keplerian_to_mee(initial_keplerian_orbit)

    # Calculate total Delta-V available from propellant (Tsiolkovsky)
    g0 = 9.80665
    m0 = cfg.satellite.initial_mass_kg
    m_prop = cfg.satellite.propellant_mass_kg
    total_delta_v = cfg.satellite.isp_s * g0 * np.log(m0 / (m0 - m_prop))  # ~656.3 m/s

    num_orbits = 100
    mee_deviations = []

    # Generate 100 random target orbits and calculate MEE deviations based on total Delta-V
    while len(mee_deviations) < num_orbits:
        target_mee = generate_random_target_orbit(
            deltav_max=total_delta_v,
            initial_orbit=initial_keplerian_orbit,
            earth_radius=cfg.orbit.earth_radius_m,
            mu=cfg.orbit.mu_m3_s2,
            verbose=False,
        )

        if not np.any(np.isnan(target_mee)):
            deviation = np.abs(target_mee - initial_mee)
            mee_deviations.append(deviation)

    mee_deviations = np.array(mee_deviations)  # Shape (100, 6)

    # Compute mean, standard deviation, and maximum
    mean_deviation = np.mean(mee_deviations, axis=0)
    std_deviation = np.std(mee_deviations, axis=0)
    max_deviation = np.max(mee_deviations, axis=0)

    mee_names = ["p (m)", "f (-)", "g (-)", "h (-)", "k (-)", "L (rad)"]

    print("=" * 70)
    print(f"Statistiche deviazione MEE su {num_orbits} orbite target casuali:")
    print(
        f"Delta-V totale disponibile per la manovra (Tsiolkovsky): {total_delta_v:.2f} m/s"
    )
    print("=" * 70)
    for name, mean_val, std_val, max_val in zip(
        mee_names, mean_deviation, std_deviation, max_deviation
    ):
        print(
            f"{name:10s} | Media: {mean_val:12.6e} | Std: {std_val:12.6e} | Max: {max_val:12.6e}"
        )

    print("\n" + "=" * 70)
    print(
        "Parametri di scaling consigliati in media (state_scales per [p, f, g, h, k]):"
    )
    print("=" * 70)
    print(f"state_scales (Media) = {[float(x) for x in mean_deviation[:5]]}")
    print(
        f"Config YAML format:\n  state_scales: [{', '.join([f'{val:.6e}' for val in mean_deviation[:5]])}]"
    )
    print("=" * 70)
