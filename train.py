from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from airspace_wrapper import AirspaceMultiAgentEnv
from multi_agent_wrapper import MultiAgentWrapper


# Wrap your environment
wrapped_env = make_vec_env(lambda: MultiAgentWrapper(AirspaceMultiAgentEnv()), n_envs=1)

# Initialize the agent
model = PPO("MlpPolicy", wrapped_env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("multi_agent_airspace_model")