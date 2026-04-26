import os

# SUMO settings
SUMO_GUI = True  # Set to True to see the simulation graphically during test/demo
SUMO_NET_FILE = os.path.join(os.path.dirname(__file__), '..', 'map.net.xml')
SUMO_ROU_FILE = os.path.join(os.path.dirname(__file__), '..', 'traffic.rou.xml')

# Simulation settings
SIM_STEP_LENGTH = 1.0     # 1 second simulation step
MAX_STEPS = 3600          # 1 hour simulation episode

# Reinforcement Learning Environment settings
OBSERVATION_SPACE_DIM = 4 # Example: 4 features per intersection node
ACTION_SPACE_DIM = 2      # Example: binary action (extend current phase or change phase)

# GAT Model settings
GAT_IN_CHANNELS = OBSERVATION_SPACE_DIM
GAT_HIDDEN_CHANNELS = 32
GAT_OUT_CHANNELS = 16 

# SB3 Training settings
TOTAL_TIMESTEPS = 50000
LEARNING_RATE = 1e-3