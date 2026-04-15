import os
import sys
import torch
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from envs.sumo_env import SumoGATEnv
from config import config
from utils import sumo_utils

def test_model():
    tls_ids = sumo_utils.get_all_tls_ids()
    if not tls_ids:
        print("ERROR: No traffic lights found in map.net.xml. Run create_map.py first.")
        return
        
    print(f"Testing on traffic lights: {tls_ids}")
    
    config.SUMO_GUI = True
    
    env = SumoGATEnv(tls_ids)
    
    model_path = "ppo_sumo_gat_model.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first by running 'python training/train.py'.")
        return
        
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env)
    
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    print("Starting evaluation. Close the SUMO window or wait for simulation to finish.")
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"Evaluation finished. Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    test_model()