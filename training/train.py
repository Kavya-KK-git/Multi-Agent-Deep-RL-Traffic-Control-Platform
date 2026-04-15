import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from envs.sumo_env import SumoGATEnv
from models.gat_model import PyGSB3FeatureExtractor
from config import config
from utils import sumo_utils

class DashboardCallback(BaseCallback):
    """
    Log metrics to a CSV for the Streamlit dashboard.
    """
    def __init__(self, verbose=0):
        super(DashboardCallback, self).__init__(verbose)
        self.log_file = "training_log.csv"
        pd.DataFrame(columns=["step", "reward", "queue_length", "avg_speed", "throughput", "total_co2", "passed_green", "passed_yellow"]).to_csv(self.log_file, index=False)
        open("signal_changes.txt", "w").close()

    def _on_step(self) -> bool:
        if self.n_calls % 10 == 0:
            reward = self.locals['rewards'][0]
            info = self.locals['infos'][0]
            queue_length = info.get("total_queue", 0)
            
            stats = sumo_utils.get_global_stats()
            avg_speed = stats.get("avg_speed", 0.0)
            throughput = stats.get("throughput", 0)
            total_co2 = stats.get("total_co2", 0.0)
            
            passed_green = info.get("passed_green", 0)
            passed_yellow = info.get("passed_yellow", 0)
            
            print(f"Step {self.num_timesteps} | Reward: {reward:.2f} | Queue: {queue_length:.1f} | Green: {passed_green} | Yellow: {passed_yellow}")
            
            new_data = pd.DataFrame(
                [[self.num_timesteps, reward, queue_length, avg_speed, throughput, total_co2, passed_green, passed_yellow]], 
                columns=["step", "reward", "queue_length", "avg_speed", "throughput", "total_co2", "passed_green", "passed_yellow"]
            )
            try:
                new_data.to_csv(self.log_file, mode='a', header=False, index=False)
            except Exception as e:
                pass
        return True

def main():
    tls_ids = sumo_utils.get_all_tls_ids()
    if not tls_ids:
        print("ERROR: No traffic lights found in map.net.xml. Run create_map.py first.")
        return
        
    print(f"Found traffic lights: {tls_ids}")
    
    num_intersections = len(tls_ids)
    if num_intersections > 1:
        edge_index = torch.tensor([
            [i for i in range(num_intersections - 1)] + [i+1 for i in range(num_intersections - 1)],
            [i+1 for i in range(num_intersections - 1)] + [i for i in range(num_intersections - 1)]
        ], dtype=torch.long)
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    
    num_intersections = len(tls_ids)
    
    env = SumoGATEnv(tls_ids)
    
   
    target_feature_dim = num_intersections * config.GAT_OUT_CHANNELS
    
    policy_kwargs = dict(
        features_extractor_class=PyGSB3FeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=target_feature_dim,
            num_intersections=num_intersections,
            obs_per_node=config.OBSERVATION_SPACE_DIM,
            edge_index=edge_index
        ),
    )
    
    print("Initializing PPO Agent with PyG Graph Extractor...")
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, 
                learning_rate=config.LEARNING_RATE, verbose=1)
    
    callback = DashboardCallback()
    print(f"Starting training for {config.TOTAL_TIMESTEPS} timesteps...")
    import traceback
    try:
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback)
    except Exception as e:
        print(f"Training interrupted or failed:")
        traceback.print_exc()
    finally:
        env.close()
        
    model.save("ppo_sumo_gat_model")
    print("Model saved to ppo_sumo_gat_model.zip")

if __name__ == "__main__":
    main()
