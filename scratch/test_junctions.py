import os
import sys
sys.path.append('.')
from envs.sumo_env import SumoGATEnv
from utils import sumo_utils

def test_env():
    print("Checking traffic lights in map...")
    tls_ids = sumo_utils.get_all_tls_ids()
    print(f"Found {len(tls_ids)} traffic lights: {tls_ids}")
    
    if len(tls_ids) > 1:
        print("SUCCESS: Multiple junctions detected!")
    else:
        print("FAILURE: Only one junction detected.")

if __name__ == "__main__":
    test_env()
