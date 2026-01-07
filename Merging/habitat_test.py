import numpy as np
import habitat_sim
from src_python.habitat_sim.agent import agent
import quaternion  
from pathlib import Path
import cv2 
import pickle
import os

from main import RealTimeGlobalSG

def get_habitat_intrinsics(width, height, fov_deg=90.0):

    fov_rad = np.deg2rad(fov_deg)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = (height / 2.0) / np.tan(fov_rad / 2.0) 
    cx = width / 2.0
    cy = height / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K

def _render(sim, path_generator, goal, rt_scene_graph, max_steps=1000):

    for step_count in range(max_steps):
        try:
            action = path_generator.next_action_along(goal)
        except habitat_sim.errors.GreedyFollowerError:
            action_list = ["turn_left", "move_forward"]
            action = np.random.choice(action_list)
            continue 

        if action is None:
            return
            
        obs = sim.step(action)        

        # RGB: Habitat returns RGBA or RGB, convert to BGR for OpenCV/internal logic if needed
        rgb_image = obs["rgb"]
        if len(rgb_image.shape) > 2:
            rgb_image = rgb_image[..., 0:3][..., ::-1] # RGB to BGR

        # Depth: Habitat returns metric depth or normalized. 
        depth_image = obs["depth"]
        metric_depth = np.clip(depth_image, 0, 10) 
        
        agent_state = sim.get_agent(0).get_state()
        
        # Quaternion to 3x3 Matrix
        q = agent_state.rotation
        rot_mat = quaternion.as_rotation_matrix(q)
        
        trans = agent_state.position 

        # Update Global Scene Graph
        try:
            global_sg_data = rt_scene_graph.process_frame(
                rgb_image, 
                metric_depth, 
                rot_mat, 
                trans
            )
            # breakpoint()
            num_objects = len(global_sg_data["pcd"])
            print(f"Step {step_count}: Global SG updated. Total objects: {num_objects}")
            
        except Exception as e:
            print(f"Error updating scene graph: {e}")
        
        cv2.imshow("monocular_rgb", rgb_image)
        
        d_vis = (metric_depth / 10.0)
        cv2.imshow("monocular_depth", d_vis)
        
        cv2.waitKey(20) 

def main(args):
    import cv2
    
    # --- Habitat Configuration ---
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = (
        "/home/theya/vln_marmot/FROSS/Datasets/Replica/data/apartment_0/habitat/mesh_semantic.ply"
    )

    # Sensors
    res_w, res_h = 512, 512
    hfov = 90.0
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.resolution = [res_h, res_w]
    rgb_sensor.position = 1.5 * habitat_sim.geo.UP 
    rgb_sensor.hfov = hfov

    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.resolution = [res_h, res_w]
    depth_sensor.position = 1.5 * habitat_sim.geo.UP
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.hfov = hfov

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [rgb_sensor, depth_sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))
    
    sim.pathfinder.load_nav_mesh(
        "/home/theya/vln_marmot/FROSS/Datasets/Replica/data/apartment_0/habitat/mesh_semantic.navmesh"
    )

    agent = sim.get_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.0, 0.0, 0.0])
    agent.set_state(agent_state)
    path_generator = habitat_sim.nav.GreedyGeodesicFollower(sim.pathfinder, agent=agent, goal_radius=0.3)

    print("Initializing Real-Time Scene Graph...")
    intrinsics = get_habitat_intrinsics(res_w, res_h)
    
    try:
        rt_scene_graph = RealTimeGlobalSG(args, intrinsics)
        print("Scene Graph Initialized successfully.")
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not load Scene Graph mappings.")
        print(f"Details: {e}")
        print(f"Checked path: {args.dataset_path}/ReplicaSSG")
        print("Please ensure 'ReplicaSSG' folder is in your dataset_path.\n")
        sim.close()
        return

    for i in range(5): 
        sim.pathfinder.seed(i)
        goal = sim.pathfinder.get_random_navigable_point()
        print(f"Episode {i}: Navigating to {goal}")
        _render(sim, path_generator, goal, rt_scene_graph)

    final_graph = rt_scene_graph.get_current_graph()
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    save_path = args.output_path / "global_scene_graph.pkl"
    
    print(f"Saving Global Scene Graph to: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(final_graph, f)
    
    print("Save complete.")
    sim.close()

if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--no-display", dest="display", action="store_false")
    args.set_defaults(display=True)
    args.add_argument("--artifact_path", type=Path, default='/home/theya/vln_marmot/FROSS/weights/RT-DETR-EGTR/VG/egtr__RT-DETR__VG__last.pth/batch__6__epochs__50_25__lr__2e-07_2e-06_2e-05__finetune/version_0', help="Path to weights.")
    
    args.add_argument("--dataset_path", type=str, default="/home/theya/vln_marmot/FROSS/Datasets/", help="Path to dataset root containing ReplicaSSG.")
    
    args.add_argument("--label_categories", type=str, choices=["scannet", "replica"], default="replica") 
    args.add_argument("--obj_thresh", type=float, default=0.5)
    args.add_argument("--rel_topk", type=int, default=20)
    
    # Flags required by SG_Predictor
    args.add_argument("--use_gt_sg", action="store_true", default=False)
    args.add_argument("--split", type=str, default="test")
    args.add_argument("--output_path", type=Path, default="output/")
    args.add_argument("--hellinger_threshold", type=float, default=0.85)
    args.add_argument("--not_use_gt_pose", action="store_true", default=True)
    args.add_argument("--kf_strategy", type=str, default="none")
    args.add_argument("--use_kim", action="store_true", default=False)
    args.add_argument("--visualize_folder", type=Path, default=None)

    parsed_args = args.parse_args()
    parsed_args.use_gt_pose = not parsed_args.not_use_gt_pose
    
    main(parsed_args)