import sumolib
import os
from config.config import SUMO_NET_FILE

def find_junction_coords():
    net = sumolib.net.readNet(SUMO_NET_FILE)
    tls_ids = [node.getID() for node in net.getNodes() if node.getType() == 'traffic_light']
    
    print(f"Total Traffic Light Junctions: {len(tls_ids)}")
    for tls_id in tls_ids:
        node = net.getNode(tls_id)
        coord = node.getCoord()
        print(f"ID: {tls_id} | Coords (x,y): {coord}")

if __name__ == "__main__":
    find_junction_coords()
