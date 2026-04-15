import traci
import sumolib
import sys
import os
from config.config import SUMO_GUI, SUMO_NET_FILE, SUMO_ROU_FILE

def start_sumo():
    """Starts the SUMO simulation using traci."""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    sumo_binary = sumolib.checkBinary('sumo-gui') if SUMO_GUI else sumolib.checkBinary('sumo')
    
    # We use traci.start rather than sumo directly
    traci_command = [sumo_binary, "-c", "sim.sumocfg"] 
    # OR we can pass net-file and route-files directly if we don't have a sim.sumocfg:
    traci_command = [
        sumo_binary,
        "-n", SUMO_NET_FILE,
        "-r", SUMO_ROU_FILE,
        "--waiting-time-memory", "10000",
        "--time-to-teleport", "-1"
    ]
    
    traci.start(traci_command)

def get_intersection_state(tls_id):
    """
    Retrieves the traffic state for a specific traffic light (intersection).
    Gets 'queue length' and 'lane occupancy' for incoming lanes.
    """
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    
    total_queue_length = 0
    total_occupancy = 0
    
    # We remove duplicates since a lane might be registered multiple times for different connections
    unique_lanes = set(lanes)
    
    for lane in unique_lanes:
        queue = traci.lane.getLastStepHaltingNumber(lane)
        occupancy = traci.lane.getLastStepOccupancy(lane)
        
        total_queue_length += queue
        total_occupancy += occupancy
        
    avg_occupancy = total_occupancy / len(unique_lanes) if unique_lanes else 0
    
    return {
        "queue_length": total_queue_length,
        "avg_occupancy": avg_occupancy
    }

def set_intersection_phase(tls_id, phase_index):
    """Sets a specific phase for the traffic light."""
    traci.trafficlight.setPhase(tls_id, phase_index)

def close_sumo():
    """Closes the simulation running in traci."""
    try:
        traci.close()
    except:
        pass

def get_all_tls_ids():
    """Returns a list of all traffic light IDs from the network file."""
    net = sumolib.net.readNet(SUMO_NET_FILE)
    tls_list = [node.getID() for node in net.getNodes() if node.getType() == 'traffic_light']
    return tls_list

def get_global_stats():
    """
    Retrieves global traffic statistics from the simulation.
    Returns estimated throughput (vehicles currently in network), average network-wide vehicle speed, and total CO2 emission.
    """
    try:
        vehicles = traci.vehicle.getIDList()
        num_vehicles = len(vehicles)
        
        total_speed = 0
        total_co2 = 0
        
        for veh_id in vehicles:
            total_speed += traci.vehicle.getSpeed(veh_id)
            total_co2 += traci.vehicle.getCO2Emission(veh_id)
            
        avg_speed = total_speed / num_vehicles if num_vehicles > 0 else 0.0
        
        return {
            "throughput": num_vehicles,
            "avg_speed": avg_speed,
            "total_co2": total_co2
        }
    except Exception:
        # Fallback if traci is not ready or fails
        return {"throughput": 0, "avg_speed": 0.0, "total_co2": 0.0}
