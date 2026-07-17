import torch
class Agent:



    def __init__(self) -> None:
        pass
        
    def select_action(self, state, actor_critic):
        """
        Selects an action based on the current state using the Actor-Critic model.
        
        :param state: The current state of the environment.
        :param actor_critic: The ActorCritic model used to select actions.
        :return: A tuple containing the selected action and its log probability.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
        mu, sigma, _ = actor_critic(state_tensor)
        
        # Create a normal distribution with the predicted mean and standard deviation
        dist = torch.distributions.Normal(mu, sigma)
        
        # Sample an action from the distribution
        action = dist.sample()
        
        # Calculate the log probability of the selected action
        log_prob = dist.log_prob(action)
        
        return action.squeeze(0).detach().numpy(), log_prob.squeeze(0)