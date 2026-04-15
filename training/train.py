import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd

# Ensure project root is in system path to find modules
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
        # Create/Reset log file
        pd.DataFrame(columns=["step", "reward", "queue_length", "avg_speed", "throughput", "total_co2"]).to_csv(self.log_file, index=False)
        # Clear previous signal changes log
        open("signal_changes.txt", "w").close()

    def _on_step(self) -> bool:
        # Log every 10 steps to keep dashboard responsive
        if self.n_calls % 10 == 0:
            reward = self.locals['rewards'][0]
            info = self.locals['infos'][0]
            queue_length = info.get("total_queue", 0)
            
            # Fetch global stats
            stats = sumo_utils.get_global_stats()
            avg_speed = stats.get("avg_speed", 0.0)
            throughput = stats.get("throughput", 0)
            total_co2 = stats.get("total_co2", 0.0)
            
            # Print to standard output so the user sees it in their output screen
            print(f"Step {self.num_timesteps} | Reward: {reward:.2f} | Queue: {queue_length:.1f} | Avg Speed: {avg_speed:.2f} m/s")
            
            new_data = pd.DataFrame(
                [[self.num_timesteps, reward, queue_length, avg_speed, throughput, total_co2]], 
                columns=["step", "reward", "queue_length", "avg_speed", "throughput", "total_co2"]
            )
            # Append safely
            try:
                new_data.to_csv(self.log_file, mode='a', header=False, index=False)
            except Exception as e:
                pass # Ignore file lock issues briefly
        return True

def main():
    # 1. AUTOMATICALLY get all traffic light IDs from the map
    tls_ids = sumo_utils.get_all_tls_ids()
    if not tls_ids:
        print("ERROR: No traffic lights found in map.net.xml. Run create_map.py first.")
        return
        
    print(f"Found traffic lights: {tls_ids}")
    
    # 2. Define the static road graph connecting the intersections
    # For GAT to work, we need an edge_index. 
    # For simplicity, if there are multiple nodes, we create a basic ring or sequential graph
    # unless we read it from the network.
    num_intersections = len(tls_ids)
    if num_intersections > 1:
        # Connect nodes in a simple line for the skeleton 
        edge_index = torch.tensor([
            [i for i in range(num_intersections - 1)] + [i+1 for i in range(num_intersections - 1)],
            [i+1 for i in range(num_intersections - 1)] + [i for i in range(num_intersections - 1)]
        ], dtype=torch.long)
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    
    num_intersections = len(tls_ids)
    
    # 3. Create Custom Gymnasium Environment
    env = SumoGATEnv(tls_ids)
    
    # Optional: Verify the environment follows Gym standards (uncomment to test)
    # check_env(env)
    
    # 4. Integrate PyG GAT model into SB3 Policy
    # We specify our custom feature extractor to be the GAT model.
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
    
    # 5. Initialize the RL Agent (PPO)
    print("Initializing PPO Agent with PyG Graph Extractor...")
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, 
                learning_rate=config.LEARNING_RATE, verbose=1)
    
    # 6. Train the Agent with the Dashboard Callback
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
        
    # 7. Save Model
    model.save("ppo_sumo_gat_model")
    print("Model saved to ppo_sumo_gat_model.zip")

if __name__ == "__main__":
    main()
