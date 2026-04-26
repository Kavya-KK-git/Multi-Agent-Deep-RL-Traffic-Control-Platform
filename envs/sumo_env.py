import sys
import os
import gymnasium as gym
import numpy as np
import traci
from config import config
from utils import sumo_utils

class SumoGATEnv(gym.Env):
    """
    A unified Gymnasium Environment for an intersection controlled by RL.
    For simplicity, this handles a single centralized controller over multiple intersections, 
    but can be converted to PettingZoo for multi-agent specific API.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, tls_ids=None):
        super(SumoGATEnv, self).__init__()
        
        self.tls_ids = tls_ids if tls_ids else []
        self.num_intersections = len(self.tls_ids) if self.tls_ids else 1 # Placeholder
        self.last_switch_step = {}
        
        # We will re-initialize spaces in reset() if tls_ids were empty
        self.action_space = gym.spaces.MultiDiscrete([config.ACTION_SPACE_DIM] * self.num_intersections)
        flat_obs_dim = self.num_intersections * config.OBSERVATION_SPACE_DIM
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(flat_obs_dim,), dtype=np.float32
        )
        
        self.step_count = 0
        self.per_junction_stats = {} # Will be {tls_id: {"green": 0, "yellow": 0, "red": 0}}
        self.prev_lane_vehs = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_lane_vehs = {}
        
        # Restart the SUMO simulation
        sumo_utils.close_sumo()
        sumo_utils.start_sumo()
        
        available_tls = traci.trafficlight.getIDList()
        if not self.tls_ids:
            self.tls_ids = list(available_tls)
            self.num_intersections = len(self.tls_ids)
            
            # Re-initialize spaces to match found TLS count
            self.action_space = gym.spaces.MultiDiscrete([config.ACTION_SPACE_DIM] * self.num_intersections)
            flat_obs_dim = self.num_intersections * config.OBSERVATION_SPACE_DIM
            self.observation_space = gym.spaces.Box(
                low=0, high=np.inf, shape=(flat_obs_dim,), dtype=np.float32
            )
            
        # Initialize stats per junction
        self.per_junction_stats = {tls_id: {"green": 0, "yellow": 0, "red": 0} for tls_id in self.tls_ids}
        self.last_switch_step = {tls_id: 0 for tls_id in self.tls_ids}
        
        if not self.tls_ids:
            print("WARNING: No Traffic Lights found in the SUMO map!")
        
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        
        # Apply actions
        for i, tls_id in enumerate(self.tls_ids):
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(logic.phases)
            
            current_phase = traci.trafficlight.getPhase(tls_id)
            if action[i] == 1:
                next_phase = (current_phase + 1) % num_phases
                sumo_utils.set_intersection_phase(tls_id, next_phase)
                duration = self.step_count - self.last_switch_step.get(tls_id, 0)
                self.last_switch_step[tls_id] = self.step_count
                
                state = sumo_utils.get_intersection_state(tls_id)
                log_msg = f"[Junction: {tls_id}] SIGNAL CHANGED | Maintained Phase for: {duration}s | Cars Waiting: {state['queue_length']}\n"
                with open("signal_changes.txt", "a") as f:
                    f.write(log_msg)
        
        for _ in range(int(config.SIM_STEP_LENGTH)):
            traci.simulationStep()
            
            for tls_id in self.tls_ids:
                state_string = traci.trafficlight.getRedYellowGreenState(tls_id)
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                
                # SIMULATE RED LIGHT VIOLATIONS (Test Case)
                # Occasionally force a vehicle to ignore the red light for monitoring purposes
                if 'r' in state_string.lower() or 'R' in state_string.lower():
                    for lane in set(controlled_lanes):
                        vehs = traci.lane.getLastStepVehicleIDs(lane)
                        if vehs and np.random.random() < 0.005: # 0.5% chance per step for testing
                            v_id = vehs[0]
                            # Bit 0: ignore safe velocity, Bit 1: ignore junction, Bit 2: ignore traffic lights
                            traci.vehicle.setSpeedMode(v_id, 0) 
                            traci.vehicle.setSpeed(v_id, 10.0) # Force move at 10m/s
                
                for idx, lane in enumerate(controlled_lanes):
                    curr_vehs = set(traci.lane.getLastStepVehicleIDs(lane))
                    prev_vehs = self.prev_lane_vehs.get(lane, set())
                    
                    # Detection of vehicle passing the stop line
                    passed_vehs = prev_vehs - curr_vehs
                    for veh_id in passed_vehs:
                        signal_state = state_string[idx]
                        if signal_state in ['G', 'g']:
                            self.per_junction_stats[tls_id]["green"] += 1
                        elif signal_state in ['y', 'Y']:
                            self.per_junction_stats[tls_id]["yellow"] += 1
                        elif signal_state in ['r', 'R']:
                            # This tracks "Red Light Violations" or late crossings
                            self.per_junction_stats[tls_id]["red"] += 1
                    
                    self.prev_lane_vehs[lane] = curr_vehs
            
        obs = self._get_observation()
        reward = self._compute_reward()
        
        total_queue = sum(obs[i*config.OBSERVATION_SPACE_DIM] for i in range(self.num_intersections))
        
        junction_queues = {}
        for i, tls_id in enumerate(self.tls_ids):
            junction_queues[tls_id] = float(obs[i * config.OBSERVATION_SPACE_DIM])
            
        terminated = self.step_count >= config.MAX_STEPS
        truncated = False
        info = {
            "total_queue": total_queue,
            "junction_queues": junction_queues,
            "per_junction_stats": self.per_junction_stats
        }
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Builds observation vector across all intersections."""
        obs = []
        for tls_id in self.tls_ids:
            # Get state using the utility function
            state = sumo_utils.get_intersection_state(tls_id)
            
            # Example vector per node [queue_length, occupancy, current_phase, 0 (padding)]
            queue = state["queue_length"]
            occ = state["avg_occupancy"]
            phase = traci.trafficlight.getPhase(tls_id)
            node_feat = [queue, occ, phase, 0.0]  # Dim: config.OBSERVATION_SPACE_DIM
            
            obs.extend(node_feat)
            
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        """
        Computes a global negative reward based on queue lengths.
        The goal is to minimize waiting vehicles.
        """
        total_queue = 0
        for tls_id in self.tls_ids:
            state = sumo_utils.get_intersection_state(tls_id)
            total_queue += state["queue_length"]
            
        return -float(total_queue)

    def close(self):
        sumo_utils.close_sumo()
