import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
env = None
def get_random_map(size: int):
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=size), render_mode="rgb_array")
    observation, info = env.reset()
    env.render()
    for _ in range(10):
        action = env.action_space.sample()  # Choose a random action
        observation, reward, terminated, truncated, info = env.step(action)
        # Render frame as RGB array
        frame = env.render()  # Returns a NumPy array (image)

        if terminated or truncated:
            break
    
def main():
    get_random_map(4)  # Example usage with a 4x4 map
    
    
if __name__ == "__main__":
    main()