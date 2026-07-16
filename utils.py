import numpy as np
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
    a = np.sin(dlat / 2)**2 + np.cos(lat) * np.cos(target_lat) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Earth's radius in meters
    R = earth_radius 
    distance = R * c
    
    return distance