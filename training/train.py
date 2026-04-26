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
    def __init__(self, tls_ids=None, verbose=0):
        super(DashboardCallback, self).__init__(verbose)
        self.log_file = "training_log.csv"
        self.tls_ids = tls_ids if tls_ids else []
        
        # Prepare columns: global metrics + per-junction stats
        columns = ["step", "reward", "queue_length", "avg_speed", "throughput", "total_co2"]
        for i in range(len(self.tls_ids)):
            columns.append(f"tls_{i}_queue")
            columns.append(f"tls_{i}_green")
            columns.append(f"tls_{i}_yellow")
            columns.append(f"tls_{i}_red")
            
        pd.DataFrame(columns=columns).to_csv(self.log_file, index=False)
        open("signal_changes.txt", "w").close()

    def _on_step(self) -> bool:
        if self.n_calls % 10 == 0:
            reward = self.locals['rewards'][0]
            info = self.locals['infos'][0]
            queue_length = info.get("total_queue", 0)
            junction_queues = info.get("junction_queues", {})
            per_junction_stats = info.get("per_junction_stats", {})
            
            stats = sumo_utils.get_global_stats()
            avg_speed = stats.get("avg_speed", 0.0)
            throughput = stats.get("throughput", 0)
            total_co2 = stats.get("total_co2", 0.0)
            
            print(f"Step {self.num_timesteps} | Reward: {reward:.2f} | Total Queue: {queue_length:.1f}")
            
            # Prepare data row
            row = [self.num_timesteps, reward, queue_length, avg_speed, throughput, total_co2]
            for tls_id in self.tls_ids:
                row.append(junction_queues.get(tls_id, 0.0))
                stats_j = per_junction_stats.get(tls_id, {"green": 0, "yellow": 0, "red": 0})
                row.append(stats_j["green"])
                row.append(stats_j["yellow"])
                row.append(stats_j["red"])
            
            new_data = pd.DataFrame([row])
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
    
    # Dynamically build the GAT edge index based on actual road connections
    graph_edges = sumo_utils.get_network_graph(tls_ids)
    if graph_edges:
        # Convert list of tuples to [2, E] tensor
        from_nodes, to_nodes = zip(*graph_edges)
        edge_index = torch.tensor([from_nodes, to_nodes], dtype=torch.long)
    else:
        # Fallback to self-loops if no connections are found
        edge_index = torch.tensor([[i for i in range(num_intersections)], 
                                    [i for i in range(num_intersections)]], dtype=torch.long)
    
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
    
    callback = DashboardCallback(tls_ids=tls_ids)
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
