import torch
import torch.optim as optim
from agent import Agent
from model import ActorCritic


state_dim = 9
action_dim = 3
model = ActorCritic(state_dim, action_dim)
agent = Agent()

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action, log_prob = agent.select_action(state, model)
        next_state, reward, done, _ = env.step(action)
        
        
        
        state = next_state