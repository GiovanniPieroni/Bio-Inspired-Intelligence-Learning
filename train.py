import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tudatpy.astro import element_conversion
from SatelliteEnv import SatelliteEnv
from model import Actor, Critic
from agent import Agent, ReplayBuffer

state_dim = 10
action_dim = 3
num_episodes = 1000
batch_size = 64
gamma = 0.99
tau = 0.005  # Incremental parameter for soft update of target networks


discount_factor = 0.99


# Initialization of principal networks
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim, action_dim)

# Initialize Target Networks (Q' and mu' in the algorithm)
target_actor = Actor(state_dim, action_dim)
target_critic = Critic(state_dim, action_dim)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

# Independent Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

agent = Agent(action_dim=action_dim, exploration_noise=0.15)
# Replay buffer for experience replay
replay_buffer = ReplayBuffer(capacity=50000)

# [QUI INSERISCI LA TUA INIZIALIZZAZIONE DELL'AMBIENTE env COME GIA' SCRITTA]
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
cartesian_state = element_conversion.keplerian_to_cartesian(
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


rewards_history = []
actor_loss_history = []
critic_loss_history = []

for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    episode_reward = 0.0
    ep_actor_loss, ep_critic_loss, steps = 0.0, 0.0, 0

    if episode % 50 == 0:
        print(f"Episode {episode}/{num_episodes}")

    while not done:
        # Seleziona l'azione con rumore
        action = agent.select_action(state, actor, add_noise=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Salva la transizione nel Replay Buffer R
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        steps += 1

        # Training only occurs if there is enough data in the buffer
        if len(replay_buffer) > batch_size:
            # Sample from Replay Buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_next_states,
                batch_dones,
            ) = replay_buffer.sample(
                batch_size
            )  # Sample a batch of experiences

            # UPDATE CRITIC (Q-Network)
            with torch.no_grad():
                # Azione futura deterministica stimata dal target actor mu'
                next_actions = target_actor(batch_next_states)
                # Future Q-value estimated by the target critic Q'
                target_Q = target_critic(batch_next_states, next_actions)
                # Calculate Bellman target y_i
                y_i = batch_rewards + gamma * target_Q * (1 - batch_dones)

            # Current value estimated by the main critic Q
            current_Q = critic(batch_states, batch_actions)
            # Minimize Mean Squared Error (Critic Loss)
            critic_loss = F.mse_loss(current_Q, y_i)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # UPDATE ACTOR (Policy Network)
            # The Actor maximizes the Q-value, so we minimize its negative (-Q).
            # This is the PyTorch translation of the mathematical Chain Rule seen in class!
            actor_actions = actor(batch_states)
            actor_loss = -critic(batch_states, actor_actions).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # SOFT UPDATE OF TARGET NETWORKS
            for param, target_param in zip(
                critic.parameters(), target_critic.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            for param, target_param in zip(
                actor.parameters(), target_actor.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            ep_actor_loss += actor_loss.item()
            ep_critic_loss += critic_loss.item()

    rewards_history.append(episode_reward)
    if steps > 0:
        actor_loss_history.append(ep_actor_loss / steps)
        critic_loss_history.append(ep_critic_loss / steps)

# Save final weights
torch.save(actor.state_dict(), "satellite_actor_DDPG.pth")
torch.save(critic.state_dict(), "satellite_critic_DDPG.pth")
print("DDPG training completed!")
