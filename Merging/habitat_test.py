import numpy as np

import habitat_sim
from src_python.habitat_sim.agent import agent
import quaternion  

from pathlib import Path
from sg_prediction import SG_Predictor
from detect import annotate_scene_graph

# import cv2
cv2 = None

# Helper function to render observations from the stereo agent
def _render(sim, path_generator, goal, sg_predictor, max_steps=1000):
    for _ in range(max_steps):
        # print("Goal:", goal)
        try:
            action = path_generator.next_action_along(goal)
        except habitat_sim.errors.GreedyFollowerError:
            # print("Recovery behavior triggered")
            action_list = ["turn_left", "move_forward"]
            action = np.random.choice(action_list)
            continue 
        # action = path_generator.next_action_along(goal)
        # for _ in range(35):
            # Just spin in a circle
            # obs = sim.step("turn_right")
        if action is None:
            return
        obs = sim.step(action)

        rgb_image = obs["rgb"]

        depth_image = obs["depth"]
        depth_image = np.clip(depth_image, 0, 10)
        depth_image /= 10.0


        # If in RGB/RGBA format, change first to RGB and change to BGR
        if len(rgb_image.shape) > 2:
            rgb_image = rgb_image[..., 0:3][..., ::-1]

        fross = annotate_scene_graph(rgb_image, sg_predictor)
        # display=False is used for the smoke test
        # if display:
        cv2.imshow("monocular_depth", depth_image)
        cv2.imshow("monocular_rgb", rgb_image)
        cv2.imshow("fross_scene_graph", fross)
        cv2.waitKey(200)

def main(args):
    global cv2
    # Only import cv2 if we are doing to display
    import cv2
    sg_predictor = SG_Predictor(args)
        # cv2.namedWindow("monocular_rgb", cv2.WINDOW_AUTOSIZE)

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = (
        "/home/theya/vln_marmot/FROSS/Datasets/Replica/data/office_3/habitat/mesh_semantic.ply"
    )

    # First, let's create a stereo RGB agent
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.resolution = [512, 512]
    rgb_sensor.position = 1.5 * habitat_sim.geo.UP 

    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.resolution = [512, 512]
    depth_sensor.position = 1.5 * habitat_sim.geo.UP
    # The only difference is that we set the sensor type to DEPTH
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [rgb_sensor, depth_sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))
    sim.pathfinder.load_nav_mesh(
    "/home/theya/vln_marmot/FROSS/Datasets/Replica/data/office_3/habitat/mesh_semantic.navmesh")

    agent = sim.get_agent(0)

    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.0, 0.0, 0.0])  # x, y, z in world frame

    agent.set_state(agent_state)
    path_generator = habitat_sim.nav.GreedyGeodesicFollower(sim.pathfinder, agent=agent, goal_radius=0.3)

    # top_down = sim.pathfinder.get_topdown_view(0.1, -1.02)
    # cv2.imshow('test',top_down.astype(np.float32))
    # cv2.waitKey(1000)
    # breakpoint()
    
    for i in range(10):
        sim.pathfinder.seed(i)
        goal = sim.pathfinder.get_random_navigable_point()
        print("Goal:", goal)
        _render(sim, path_generator, goal, sg_predictor)

    sim.close()

if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--no-display", dest="display", action="store_false")
    args.set_defaults(display=True)
    args.add_argument("--artifact_path", type=Path, default='/home/theya/vln_marmot/FROSS/weights/RT-DETR-EGTR/3RScan20/egtr__RT-DETR__3RScan20__last.pth/batch__6__epochs__50_25__lr__2e-07_2e-06_0.0002__finetune/version_0', help="Path to the directory containing model checkpoints/weights.")
    args.add_argument("--dataset_path", type=str, default="./", help="Path to dataset root.")
    
    # Inference parameters
    args.add_argument("--label_categories", type=str, choices=["scannet", "replica"], default="scannet")
    args.add_argument("--obj_thresh", type=float, default=0.5, help="Confidence threshold for showing objects.")
    args.add_argument("--rel_topk", type=int, default=20)
    
    # Flags required by SG_Predictor initialization
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
