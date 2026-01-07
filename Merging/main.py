import argparse
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
# from pycocotools.coco import COCO

from sg_loader import SG_Loader, GT_SG_Loader
from keyframe_selection import PeriodicKeyframeSelector, SpatialKeyframeSelector, DynamicKeyframeSelector
from sg_prediction import SG_Predictor
from global_sg import GlobalSG_Gaussian

from detect import draw_bbox, draw_rel
class SimpleIntrinsics:

    def __init__(self, matrix, width=512, height=512):
        self.fx = float(matrix[0, 0])
        self.fy = float(matrix[1, 1])
        self.cx = float(matrix[0, 2])
        self.cy = float(matrix[1, 2])
        self.width = width
        self.height = height
        # Keep the raw matrix just in case
        self.intrinsic_matrix = matrix

class RealTimeGlobalSG:

    def __init__(self, args, intrinsic_matrix):
        self.args = args
        
        self.intrinsic_object = SimpleIntrinsics(intrinsic_matrix, width=512, height=512)
        
        split = args.split
        if args.label_categories == "scannet":
            SSG_path = f"3DSSG_subset"
            class_mapping_path = f"{SSG_path}/3dssg_to_scannet.json"
            OBJ_CLASS_NAME = "ScanNet_list"
            REL_CLASS_NAME = "ScanNet_rel"
        elif args.label_categories == "replica":
            SSG_path = f"ReplicaSSG"
            class_mapping_path = f"/home/theya/vln_marmot/FROSS/Datasets/Replica/ReplicaSSG/replica_to_visual_genome.json"
            OBJ_CLASS_NAME = "VisualGenome_list"
            REL_CLASS_NAME = "VisualGenome_rel"
        
        mapping_full_path = Path(args.dataset_path) / class_mapping_path
        
        if not mapping_full_path.exists():
             raise FileNotFoundError(
                f"\n[Error] Could not find class mapping file: {mapping_full_path}\n"
                f"Please ensure 'ReplicaSSG' folder exists inside: {args.dataset_path}"
            )

        with open(mapping_full_path) as f:
            class_mapping = json.load(f)
        
        self.obj_classes = class_mapping[OBJ_CLASS_NAME]
        self.rel_classes = class_mapping[REL_CLASS_NAME]

        self.sg_predictor = SG_Predictor(args)

        if not args.use_kim:
            self.global_sg = GlobalSG_Gaussian(
                args.hellinger_threshold, 
                len(self.obj_classes), 
                len(self.rel_classes), 
                args.visualize_folder is not None
            )
        else:
            from global_sg_kim import GlobalSG_Kim
            self.global_sg = GlobalSG_Kim(
                args.hellinger_threshold, 
                len(self.obj_classes), 
                len(self.rel_classes)
            )

    def process_frame(self, rgb_image, depth_image, camera_rot, camera_trans):

        obj_det_output, all_scores, classes, class_probs, bboxes = self.sg_predictor.detect_objects(rgb_image)
        
        rels, relation_classes = self.sg_predictor.extract_relations(obj_det_output, all_scores)

        input_classes = class_probs if self.args.use_kim else classes
        
        if self.args.use_kim:
            self.global_sg.update(
                input_classes, 
                bboxes, 
                rels, 
                relation_classes, 
                depth_image, 
                camera_rot, 
                camera_trans, 
                self.intrinsic_object,  
                rgb_image 
            )
        else:
            # Standard Gaussian update (does NOT take img)
            self.global_sg.update(
                input_classes, 
                bboxes, 
                rels, 
                relation_classes, 
                depth_image, 
                camera_rot, 
                camera_trans, 
                self.intrinsic_object 
            )
        # breakpoint()
        fig, ax = plt.subplots(1, 1)
        ax.imshow(rgb_image)

        if len(rels) > 0:
            draw_rel(
                ax,
                rels,
                relation_classes,
                bboxes,
                self.rel_classes,
                np.random.rand(len(self.rel_classes), 3),
            )

        if len(bboxes) > 0:
            draw_bbox(
                ax,
                bboxes,
                classes,
                class_probs,
                self.obj_classes,
                np.random.rand(len(self.obj_classes), 3),
            )

        ax.axis("off")
        fig.set_size_inches(
            rgb_image.shape[1] / 100,
            rgb_image.shape[0] / 100,
        )
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        annotated = buf.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        plt.close(fig)
        cv2.imshow("Detections", annotated)

        return self.get_current_graph()

    def get_current_graph(self):
        """Extracts the formatted graph data from the internal GlobalSG object."""
        prediction = {}
        
        classes = self.global_sg.global_group.classes
        means = self.global_sg.global_group.means
        covs = self.global_sg.global_group.covs
        rels = self.global_sg.global_group.rels
        pcd = self.global_sg.global_group.pcd
        
        # Format Point Clouds
        point_clouds = []
        for idx in range(classes.shape[0]):
            pred_points = pcd[idx]
            if self.args.use_kim:
                pred_points = pred_points[::2500] 
            point_clouds.append(pred_points)

        prediction["pcd"] = point_clouds
        prediction["cls"] = classes
        prediction["mean"] = means
        prediction["cov"] = covs
        
        # Format Edges
        s, o = np.nonzero(np.sum(rels, axis=-1))
        prediction["edge_index"] = np.array([s, o])
        if len(s) > 0:
            prediction["edge_cls"] = np.array(rels[s, o] / np.sum(rels[s, o], axis=-1, keepdims=True))
        else:
            prediction["edge_cls"] = np.array([])

        return prediction

def main(args):
    pass 

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--artifact_path", type=Path)
    args.add_argument("--output_path", type=Path, default="output/")
    args.add_argument("--label_categories", type=str, choices=["scannet", "replica"], default="replica")
    args.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    args.add_argument("--obj_thresh", type=float, default=0.7)
    args.add_argument("--rel_topk", type=int, default=10)
    args.add_argument("--hellinger_threshold", type=float, default=0.85)
    args.add_argument("--use_gt_sg", action="store_true", default=False)
    args.add_argument("--not_use_gt_pose", action="store_true", default=False)
    args.add_argument("--not_preload", action="store_true", default=False)
    args.add_argument("--visualize_folder", type=Path, default=None)
    args.add_argument("--kf_strategy", type=str, default="none")
    args.add_argument("--kf_interval", type=int, default=1)
    args.add_argument("--kf_translation", type=float, default=0.01)
    args.add_argument("--kf_rotation", type=float, default=0.017)
    args.add_argument("--kf_iou_thresh", type=float, default=0.2)
    args.add_argument("--use_kim", action="store_true", default=False)
    
    args = args.parse_args()
    args.use_gt_pose = not args.not_use_gt_pose
    args.preload = not args.not_preload
    main(args)