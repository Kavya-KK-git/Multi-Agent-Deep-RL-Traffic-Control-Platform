import os
import subprocess
import urllib.request

def download_osm_map(bbox, filename="salem_map.osm"):
    """Downloads a real map from OpenStreetMap based on a bounding box."""
    # Bounding box coordinates
    min_lat, min_lon, max_lat, max_lon = bbox
    
    print(f"Downloading Salem Real Map Data from OpenStreetMap...")
    print(f"Location: Five Roads Junction, Salem (BBox: {bbox})")
    
    # Use main OSM API instead of Overpass to avoid timeouts
    osm_url = f"https://api.openstreetmap.org/api/0.6/map?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
    
    try:
        urllib.request.urlretrieve(osm_url, filename)
        print(f"[SUCCESS] Successfully downloaded real dataset: {filename}\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download map: {e}")
        return False

def generate_sumo_network(osm_file="salem_map.osm", net_file="map.net.xml"):
    """Converts the OSM map into a SUMO network simulation format."""
    print("Converting OpenStreetMap to SUMO Network...")
    try:
        # We run netconvert to extract roads, junctions, and traffic lights
        subprocess.run([
            "netconvert", 
            "--osm-files", osm_file, 
            "-o", net_file,
            "--geometry.remove", "true",
            "--roundabouts.guess", "true",
            "--ramps.guess", "true",
            "--junctions.join", "true",
            "--tls.guess", "true",
            "--tls.guess-signals", "true",
            "--tls.join", "true",
            "--tls.set", "3622848273",
            "--tls.default-type", "actuated"
        ], check=True)
        print(f"[SUCCESS] Successfully created network file: {net_file}\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to convert network: {e}")
        print("Please ensure SUMO is installed and its 'bin' folder is added to PATH.")
        return False

def generate_traffic_routes(net_file="map.net.xml", route_file="traffic.rou.xml"):
    """Generates random real-time vehicle routes through the map."""
    print("Generating simulated vehicle traffic for Salem map...")
    
    if 'SUMO_HOME' not in os.environ:
        print("[ERROR] Error: SUMO_HOME environment variable is not set. Cannot run randomTrips.py")
        return False
        
    random_trips_script = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    
    try:
        # Generate continuous random trips for 1 hour (3600 seconds)
        # -p 1.5 means roughly one new distinct vehicle every 1.5 seconds
        subprocess.run([
            "python", random_trips_script,
            "-n", net_file,
            "-r", route_file,
            "-e", "3600",
            "-p", "1.5",
            "--vehicle-class", "passenger",
            "--vclass", "passenger",
            "--prefix", "veh",
            "--min-distance", "50", # Ensure trips aren't too short
            "--trip-attributes", 'departLane="best" departSpeed="max" departPos="random"'
        ], check=True)
        print(f"[SUCCESS] Successfully generated traffic routes: {route_file}\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to generate traffic: {e}")
        return False

def setup_salem_map():
    # Bounding box for a major traffic junction in Salem: Five Roads Junction
    # Using roughly (Lat: 11.670, Lon: 78.138) to (Lat: 11.677, Lon: 78.146)
    salem_bbox = (11.6700, 78.1380, 11.6770, 78.1460)
    
    success = download_osm_map(salem_bbox)
    if success:
        success = generate_sumo_network()
        if success:
            generate_traffic_routes()
            print("🚀 REAL DATASET PREPARATION COMPLETE!")
            print("The Salem network is now ready. You can start/restart your Streamlit dashboard and begin training the AI on Salem roads.")

if __name__ == "__main__":
    setup_salem_map()
