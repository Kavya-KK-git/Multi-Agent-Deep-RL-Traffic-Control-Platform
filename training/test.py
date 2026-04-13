import os
import sys
import torch
from stable_baselines3 import PPO

# Ensure project root is in system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from envs.sumo_env import SumoGATEnv
from config import config
from utils import sumo_utils

def test_model():
    # 1. AUTOMATICALLY find the IDs used in the map
    tls_ids = sumo_utils.get_all_tls_ids()
    if not tls_ids:
        print("ERROR: No traffic lights found in map.net.xml. Run create_map.py first.")
        return
        
    print(f"Testing on traffic lights: {tls_ids}")
    
    # 2. Enable GUI for testing so user can see it
    config.SUMO_GUI = True
    
    # 3. Initialize the Environment with the same IDs
    env = SumoGATEnv(tls_ids)
    
    # 4. Load the trained model
    model_path = "ppo_sumo_gat_model.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first by running 'python training/train.py'.")
        return
        
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env)
    
    # 5. Run test episode
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    print("Starting evaluation. Close the SUMO window or wait for simulation to finish.")
    
    while not (done or truncated):
        # Predict action using the trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"Evaluation finished. Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    test_model()