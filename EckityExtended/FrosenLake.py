import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
env = None
def get_random_map(size: int):
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=size), render_mode="rgb_array")
    observation, info = env.reset()
    desc = env.unwrapped.desc.astype(str)  # This gives you the grid layout
    print("Random Frozen Lake Map:")
    print(desc)
    for _ in range(10):
        action = env.action_space.sample()  # Choose a random action
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        # Render frame as RGB array
        frame = env.render()
        plt.imshow(frame)
        plt.axis('off')
        if terminated or truncated:
            break
    
def main():
    get_random_map(4)  # Example usage with a 4x4 map
    
    
if __name__ == "__main__":
    main()