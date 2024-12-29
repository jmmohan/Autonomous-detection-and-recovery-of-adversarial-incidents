#### recovery_agent.py

from stable_baselines3 import DQN
from rl_env import RecoveryEnv

def train_recovery_agent():
    env = RecoveryEnv()
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    return model
