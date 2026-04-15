import subprocess
import os

def create_sumo_files():
    print("Generating SUMO network using netconvert...")
    
    with open("map.nod.xml", "w") as f:
        f.write("""<nodes>
    <node id="center" x="0" y="0" type="traffic_light"/>
    <node id="top" x="0" y="100" type="priority"/>
    <node id="bottom" x="0" y="-100" type="priority"/>
    <node id="left" x="-100" y="0" type="priority"/>
    <node id="right" x="100" y="0" type="priority"/>
</nodes>""")

    with open("map.edg.xml", "w") as f:
        f.write("""<edges>
    <edge id="t2c" from="top" to="center" priority="1" numLanes="1" speed="13.89"/>
    <edge id="b2c" from="bottom" to="center" priority="1" numLanes="1" speed="13.89"/>
    <edge id="l2c" from="left" to="center" priority="1" numLanes="1" speed="13.89"/>
    <edge id="r2c" from="right" to="center" priority="1" numLanes="1" speed="13.89"/>
    <edge id="c2t" from="center" to="top" priority="1" numLanes="1" speed="13.89"/>
    <edge id="c2b" from="center" to="bottom" priority="1" numLanes="1" speed="13.89"/>
    <edge id="c2l" from="center" to="left" priority="1" numLanes="1" speed="13.89"/>
    <edge id="c2r" from="center" to="right" priority="1" numLanes="1" speed="13.89"/>
</edges>""")

    try:
        subprocess.run(["netconvert", "--node-files=map.nod.xml", "--edge-files=map.edg.xml", "--output-file=map.net.xml"], check=True)
        print("Successfully created map.net.xml")
    except Exception as e:
        print(f"Error: {e}. Make sure SUMO 'bin' is in your PATH.")

    with open("traffic.rou.xml", "w") as f:
        f.write("""<routes>
    <vType id="car" accel="1.0" decel="4.5" length="5.0" maxSpeed="15.0" />
    <flow id="f1" type="car" begin="0" end="3600" probability="0.3" from="t2c" to="c2b"/>
    <flow id="f2" type="car" begin="0" end="3600" probability="0.3" from="b2c" to="c2t"/>
</routes>""")
    print("Successfully created traffic.rou.xml")

if __name__ == "__main__":
    create_sumo_files()
