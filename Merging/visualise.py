# import pickle
# import argparse
# import open3d as o3d
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering
# import random
# import numpy as np

# # --- CLASS MAPPING ---
# CLASS_NAMES = {
#     0: "undefined", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
#     6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf", 11: "picture",
#     12: "counter", 13: "blinds", 14: "desk", 15: "shelves", 16: "curtain",
#     17: "dresser", 18: "pillow", 19: "mirror", 20: "floor_mat", 21: "clothes",
#     22: "ceiling", 23: "books", 24: "refridgerator", 25: "television", 26: "paper",
#     27: "towel", 28: "shower_curtain", 29: "box", 30: "whiteboard", 31: "person",
#     32: "night_stand", 33: "toilet", 34: "sink", 35: "lamp", 36: "bathtub",
#     37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
# }

# def get_label(class_id):
#     return CLASS_NAMES.get(int(class_id), f"Obj_{int(class_id)}")

# def main(file_path):
#     print(f"Loading {file_path}...")
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)

#     # breakpoint()
#     # Initialize the Modern GUI Application
#     app = gui.Application.instance
#     app.initialize()
    
#     # Create the window
#     w = app.create_window("FROSS Scene Graph Visualization", 1280, 720)
#     widget3d = gui.SceneWidget()
#     widget3d.scene = rendering.Open3DScene(w.renderer)
#     w.add_child(widget3d)

#     # Extract data
#     point_clouds = data['pcd']
#     # classes = data['cls']
    
#     # Material for the point clouds (Unlit allows colors to shine without lights)
#     mat = rendering.MaterialRecord()
#     mat.shader = "defaultUnlit"
#     mat.point_size = 4.0

#     print(f"Processing {len(point_clouds)} objects...")

#     # pcd = o3d.io.read_point_cloud("cloud.pcd")
#     import matplotlib.pyplot as plt
#     pcds = []  # list of PointClouds
#     cmap = plt.get_cmap("tab20")  # good for up to 20 objects

#     for idx, i in enumerate(point_clouds):
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(i)

#         color = cmap(idx % 20)[:3]  # RGB
#         pcd.paint_uniform_color(color)

#         pcds.append(pcd)

#     o3d.visualization.draw_geometries(pcds)

#     # for i, points in enumerate(point_clouds):
#     #     if len(points) < 50: continue # Skip noise
        
#     #     # 1. Prepare Geometry
#     #     pcd = o3d.geometry.PointCloud()
#     #     pcd.points = o3d.utility.Vector3dVector(points)
        
#     #     # Consistent coloring based on class ID
#     #     random.seed(int(classes[i]))
#     #     color = [random.random(), random.random(), random.random()]
#     #     pcd.paint_uniform_color(color)
        
#     #     # 2. Bounding Box & Stats
#     #     try:
#     #         aabb = pcd.get_axis_aligned_bounding_box()
#     #         aabb.color = color
#     #         extent = aabb.get_extent()
#     #         volume = extent[0] * extent[1] * extent[2]
#     #         center = aabb.get_center()

#     #         # Filter massive walls/floors
#     #         if volume > 15.0: continue

#     #         # 3. Add to Scene
#     #         # We give each object a unique name string
#     #         obj_name = f"obj_{i}_{get_label(classes[i])}"
            
#     #         # Add Point Cloud
#     #         widget3d.scene.add_geometry(obj_name + "_pcd", pcd, mat)
            
#     #         # Add Bounding Box (Lines)
#     #         # Create a line set from the AABB for visualization
#     #         lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
#     #         lines.paint_uniform_color(color)
#     #         widget3d.scene.add_geometry(obj_name + "_box", lines, mat)
            
#     #         # 4. Add Text Label
#     #         # Position the text slightly above the box
#     #         text_pos = [center[0], center[1] + (extent[1]/2) + 0.2, center[2]]
#     #         label_text = f"{get_label(classes[i])} ({i})"
            
#     #         # Add 3D Label
#     #         widget3d.add_3d_label(text_pos, label_text)

#     #     except Exception as e:
#     #         print(f"Skipping object {i}: {e}")

#     # # Add Coordinate Frame
#     # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
#     # widget3d.scene.add_geometry("axes", axes, mat)

#     # # Camera setup (Look at the center of the scene approximately)
#     # bbox = widget3d.scene.bounding_box
#     # widget3d.setup_camera(60.0, bbox, bbox.get_center())

#     # print("Opening Visualizer with Labels...")
#     # app.run()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--file", type=str, default="/home/theya/vln_marmot/FROSS/output/global_scene_graph.pkl")
#     args = parser.parse_args()

#     main(args.file)

import pickle
import argparse
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import random
import numpy as np

# --- CLASS MAPPING ---
CLASS_NAMES = {
    0: "undefined", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf", 11: "picture",
    12: "counter", 13: "blinds", 14: "desk", 15: "shelves", 16: "curtain",
    17: "dresser", 18: "pillow", 19: "mirror", 20: "floor_mat", 21: "clothes",
    22: "ceiling", 23: "books", 24: "refridgerator", 25: "television", 26: "paper",
    27: "towel", 28: "shower_curtain", 29: "box", 30: "whiteboard", 31: "person",
    32: "night_stand", 33: "toilet", 34: "sink", 35: "lamp", 36: "bathtub",
    37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
}

def get_label(class_id):
    return CLASS_NAMES.get(int(class_id), f"Obj_{int(class_id)}")

def main(file_path):
    print(f"Loading {file_path}...")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Initialize the Modern GUI Application
    app = gui.Application.instance
    app.initialize()
    
    # Create the window
    w = app.create_window("FROSS Scene Graph Visualization", 1280, 720)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    w.add_child(widget3d)

    # Extract data
    point_clouds = data['pcd']
    classes = data['cls']
    
    # Material for the point clouds (Unlit allows colors to shine without lights)
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 4.0

    print(f"Processing {len(point_clouds)} objects...")

    for i, points in enumerate(point_clouds):
        if len(points) < 50: continue # Skip noise
        
        # 1. Prepare Geometry
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Consistent coloring based on class ID
        random.seed(int(classes[i]))
        color = [random.random(), random.random(), random.random()]
        pcd.paint_uniform_color(color)
        
        # 2. Bounding Box & Stats
        try:
            aabb = pcd.get_axis_aligned_bounding_box()
            aabb.color = color
            extent = aabb.get_extent()
            volume = extent[0] * extent[1] * extent[2]
            center = aabb.get_center()

            # Filter massive walls/floors
            if volume > 15.0: continue

            # 3. Add to Scene
            # We give each object a unique name string
            obj_name = f"obj_{i}_{get_label(classes[i])}"
            
            # Add Point Cloud
            widget3d.scene.add_geometry(obj_name + "_pcd", pcd, mat)
            
            # Add Bounding Box (Lines)
            # Create a line set from the AABB for visualization
            lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
            lines.paint_uniform_color(color)
            widget3d.scene.add_geometry(obj_name + "_box", lines, mat)
            
            # 4. Add Text Label
            # Position the text slightly above the box
            text_pos = [center[0], center[1] + (extent[1]/2) + 0.2, center[2]]
            label_text = f"{get_label(classes[i])} ({i})"
            
            # Add 3D Label
            widget3d.add_3d_label(text_pos, label_text)

        except Exception as e:
            print(f"Skipping object {i}: {e}")

    # Add Coordinate Frame
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    widget3d.scene.add_geometry("axes", axes, mat)

    # Camera setup (Look at the center of the scene approximately)
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())

    print("Opening Visualizer with Labels...")
    app.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="output/global_scene_graph.pkl")
    args = parser.parse_args()

    main(args.file)