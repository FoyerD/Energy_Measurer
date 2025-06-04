import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
env = None
def get_random_map(size: int):
    env = gym.make('FrozenLake-v1', desc=['SF', 'FG'], render_mode='human', is_slippery=False)
    print("Random Frozen Lake Environment Created")
    observation, info = env.reset()
    desc = env.unwrapped.desc.astype(str)  # This gives you the grid layout
    print("Random Frozen Lake Map:")
    print(desc)
    print(env.action_space)
    actions = [1, 2]
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    
def main():
    get_random_map(8)  # Example usage with a 4x4 map
    
    
if __name__ == "__main__":
    main()