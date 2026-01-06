import cv2
import argparse
import numpy as np
from pathlib import Path
import matplotlib
# Force headless backend (prevents Qt/XCB errors)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict
import sys

# Import the predictor directly
from sg_prediction import SG_Predictor

# VISUALIZATION CONSTANTS
SCANNET_CLASSES = [
    "bathtub", "bed", "bookshelf", "cabinet", "chair", "counter", "curtain",
    "desk", "door", "floor", "otherfurniture", "picture", "refridgerator",
    "shower curtain", "sink", "sofa", "table", "toilet", "wall", "window"
]
SCANNET_REL_CLASSES = [
    "attached to", "build in", "connected to", "hanging on", "part of",
    "standing on", "supported by"
]

# HELPER FUNCTIONS
np.random.seed(42)
def get_colors(n):
    return np.random.rand(n, 3)

def to_numpy(x):
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def curvature_seq(k, base):
    seq = []
    m = 2
    sign = -1
    while len(seq) < k:
        seq.append(sign * m * base)
        sign *= -1
        if sign == -1:
            m += 2
    return seq

def arc3_midpoint(p0, p1, rad):
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    chord = p1 - p0
    d = np.linalg.norm(chord)
    if d == 0: return p0
    mid = (p0 + p1) / 2
    if rad == 0: return mid
    perp = np.array([-chord[1], chord[0]]) / d
    offset = rad * d / 2
    return mid + perp * offset

# DRAWING FUNCTIONS

def draw_bbox(axe, bboxes, labels, probs, class_names, obj_colors):
    """Draws bounding boxes and labels on the matplotlib axis."""
    bboxes = to_numpy(bboxes)
    labels = to_numpy(labels)
    probs = to_numpy(probs)

    n_boxes = len(bboxes)
    for i in range(n_boxes):
        cx, cy, w, h = bboxes[i]
        x1, y1 = cx - w/2, cy - h/2
        
        l = int(labels[i])
        if l < 0 or l >= len(class_names):
             class_name = f"ID {l}"
             color = [1, 0, 0] # Default red for unknown
        else:
             class_name = class_names[l]
             color = obj_colors[l]
            
        score = np.max(probs[i]) if hasattr(probs[i], 'max') else probs[i]
        L = 0.2126*(color[0]**2.2) + 0.7152*(color[1]**2.2) + 0.0722*(color[2]**2.2)
        
        # Draw Box
        axe.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=3))
        
        # Draw Label text
        text_label = f"{class_name}: {score:.2f}"
        axe.text(x1, y1,
                 text_label, color="black" if L > 0.6 else "white",
                 fontsize=8, fontfamily="sans-serif", ha='left', va='bottom',
                 bbox=dict(boxstyle="square,pad=0.1",
                           facecolor=color, edgecolor=color, alpha=1))

def draw_rel(axe, rel_pairs, rel_labels, bboxes, rel_class_names, rel_colors):

    bboxes = to_numpy(bboxes)
    base_curvature = 0.2
    grouped = defaultdict(list)
    for r, l in zip(rel_pairs, rel_labels):
        grouped[(r[0], r[1])].append(l)

    for (si, oi), rels in grouped.items():
        curvs = curvature_seq(len(rels), base_curvature)
        for idx, rel_idx in enumerate(rels):
            if si >= len(bboxes) or oi >= len(bboxes): continue
            
            sx, sy = bboxes[si][0], bboxes[si][1]
            ox, oy = bboxes[oi][0], bboxes[oi][1]
            curvature = curvs[idx]
            
            if rel_idx < 0 or rel_idx >= len(rel_class_names): continue
            rel_color = rel_colors[rel_idx]
            rel_name = rel_class_names[rel_idx]

            arrow = FancyArrowPatch(
                (sx, sy), (ox, oy),
                connectionstyle=f"arc3,rad={curvature}",
                arrowstyle='-|>',
                mutation_scale=15, linewidth=2, color=rel_color, alpha=0.8
            )
            axe.add_patch(arrow)
            lx, ly = arc3_midpoint((sx, sy), (ox, oy), curvature)
            L = 0.2126*(rel_color[0]**2.2) + 0.7152*(rel_color[1]**2.2) + 0.0722*(rel_color[2]**2.2)
            axe.text(
                lx, ly, rel_name, ha="center", va="center", fontsize=7, 
                color="black" if L > 0.55 else "white", weight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor=rel_color, edgecolor=rel_color, alpha=0.9)
            )

def nms_xywh(boxes, scores, iou_thresh=0.5):
    """
    boxes: Nx4 in (cx, cy, w, h)
    scores: N
    returns: indices to keep
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # Convert cx,cy,w,h → x1,y1,x2,y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < iou_thresh)[0]
        order = order[inds + 1]

    return keep


def annotate_scene_graph(
    image_rgb,
    sg_predictor,
    label_categories="scannet",
    obj_thresh=0.8,
    rel_score_thresh=0.8,
):
    """
    Args:
        image_rgb (np.ndarray): HxWx3 RGB image (uint8)
        sg_predictor (SG_Predictor): initialized predictor
        label_categories (str): "scannet" or "replica"

    Returns:
        annotated_rgb (np.ndarray): HxWx3 RGB image with annotations
    """

    # Select class names
    # if label_categories == "scannet":
    VIS_OBJ_CLASSES = SCANNET_CLASSES
    VIS_REL_CLASSES = SCANNET_REL_CLASSES

    OBJ_COLORS = get_colors(len(VIS_OBJ_CLASSES) + 10)
    REL_COLORS = get_colors(len(VIS_REL_CLASSES) + 10)

    # Inference
    obj_det_output, obj_scores, pred_classes, pred_probs, pred_boxes = \
        sg_predictor.detect_objects(image_rgb)

    rels_matrix, rel_pred_classes = \
        sg_predictor.extract_relations(obj_det_output, obj_scores)
    
# Object filtering + NMS 
    boxes_all = to_numpy(pred_boxes)
    classes_all = to_numpy(pred_classes)
    probs_all = to_numpy(pred_probs)

    # Object confidence score per box
    scores_all = probs_all.max(axis=1) if probs_all.ndim > 1 else probs_all

    # Threshold
    keep_mask = scores_all >= obj_thresh
    keep_indices = np.where(keep_mask)[0]

    boxes_thr = boxes_all[keep_mask]
    classes_thr = classes_all[keep_mask]
    probs_thr = probs_all[keep_mask]
    scores_thr = scores_all[keep_mask]

    # NMS
    nms_keep = nms_xywh(boxes_thr, scores_thr, iou_thresh=0.5)

    # Final kept indices (original indexing)
    final_indices = keep_indices[nms_keep]

    # Final objects
    boxes = boxes_all[final_indices]
    classes = classes_all[final_indices]
    probs = probs_all[final_indices]
    scores = scores_all[final_indices]

    # Mapping: original index → new index
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(final_indices)}


    rel_pairs = []
    rel_labels = []

    if rels_matrix.ndim == 3:
        s_idx, o_idx = np.nonzero(np.sum(rels_matrix, axis=-1))
    elif rels_matrix.ndim == 2:
        s_idx, o_idx = np.nonzero(rels_matrix)
    else:
        return image_rgb.copy()

    for si, oi in zip(s_idx, o_idx):
        if si == oi:
            continue

        # Skip relations involving removed boxes
        if si not in old_to_new or oi not in old_to_new:
            continue

        if rels_matrix.ndim == 3:
            rel_scores = rels_matrix[si, oi]
            r_idx = np.argmax(rel_scores)
            score = rel_scores[r_idx]
        else:
            r_idx = int(rels_matrix[si, oi])
            score = 1.0

        if score > rel_score_thresh and r_idx > 0:
            rel_pairs.append([old_to_new[si], old_to_new[oi]])
            rel_labels.append(r_idx)

    # Visualization
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_rgb)

    if len(rel_pairs) > 0:
        draw_rel(
            ax,
            rel_pairs,
            rel_labels,
            boxes,
            VIS_REL_CLASSES,
            REL_COLORS,
        )

    if len(boxes) > 0:
        draw_bbox(
            ax,
            boxes,
            classes,
            probs,
            VIS_OBJ_CLASSES,
            OBJ_COLORS,
        )

    ax.axis("off")
    fig.set_size_inches(
        image_rgb.shape[1] / 100,
        image_rgb.shape[0] / 100,
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated = buf.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )

    plt.close(fig)

    return annotated


def main(args):
    # -----------------------------------------
    image_path = "/home/theya/vln_marmot/FROSS/rgb_0.jpg" 
    output_vis_path = "visualized_output.png"
    # -----------------------------------------

    # Load and Preprocess Image 
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not read image at {image_path}. Check the path.")
        sys.exit(1)
        
    # Convert BGR to RGB for the model and matplotlib
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Prepare Visualization Classes/Colors
    if args.label_categories == "scannet":
        VIS_OBJ_CLASSES = SCANNET_CLASSES
        VIS_REL_CLASSES = SCANNET_REL_CLASSES
    else:
        # Fallback/Placeholder
        print("Warning: Replica classes selected but not hardcoded in script. Using ScanNet mappings.")
        VIS_OBJ_CLASSES = SCANNET_CLASSES
        VIS_REL_CLASSES = SCANNET_REL_CLASSES

    OBJ_COLORS = get_colors(len(VIS_OBJ_CLASSES) + 10) 
    REL_COLORS = get_colors(len(VIS_REL_CLASSES) + 10)

    # Inference
    print(f"Initializing SG_Predictor with artifact path: {args.artifact_path}")
    sg_predictor = SG_Predictor(args)
    
    print("Running object detection...")
    obj_det_output, obj_scores, pred_classes, pred_probs, pred_boxes = sg_predictor.detect_objects(image_rgb)
    
    print(f"Found {len(pred_classes)} objects. Running relation extraction...")
    rels_matrix, rel_pred_classes = sg_predictor.extract_relations(obj_det_output, obj_scores)

    print("Inference finished successfully.")

    # Process Results for Visualization 
    rel_pairs_for_vis = []
    rel_labels_for_vis = []

    # Handle 2D vs 3D output for relations
    if rels_matrix.ndim == 3:
        # Standard case: (N, N, C) - Probability volume
        s_indices, o_indices = np.nonzero(np.sum(rels_matrix, axis=-1))
    elif rels_matrix.ndim == 2:
        # Compressed case: (N, N) - Class IDs or single scores
        s_indices, o_indices = np.nonzero(rels_matrix)
    else:
        print(f"Error: rels_matrix has unexpected shape {rels_matrix.shape}")
        sys.exit(1)

    for s_idx, o_idx in zip(s_indices, o_indices):
        if s_idx == o_idx: continue # Ignore self-loops

        if rels_matrix.ndim == 3:
            rel_scores_vec = rels_matrix[s_idx, o_idx]
            r_idx = np.argmax(rel_scores_vec)
            score = rel_scores_vec[r_idx]
        else:
            val = rels_matrix[s_idx, o_idx]
            r_idx = int(val)
            score = 1.0 # Assume valid if present in 2D nonzero check

        # Thresholding (adjust if your 'no relation' class is not 0)
        if score > 0.1 and r_idx > 0: 
            rel_pairs_for_vis.append([s_idx, o_idx])
            rel_labels_for_vis.append(r_idx)

    # Generate and Save Visualization
    print(f"Generating visualization...")
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_rgb)

    # Draw relations first (so boxes appear on top)
    if len(rel_pairs_for_vis) > 0:
        draw_rel(ax, rel_pairs_for_vis, rel_labels_for_vis, pred_boxes, 
                 VIS_REL_CLASSES, REL_COLORS)

    # Draw object bounding boxes
    if len(pred_boxes) > 0:
        draw_bbox(ax, pred_boxes, pred_classes, pred_probs, 
                  VIS_OBJ_CLASSES, OBJ_COLORS)

    ax.axis('off')
    # Set figure size based on image dimensions
    fig.set_size_inches(image_rgb.shape[1] / 100, image_rgb.shape[0] / 100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig(output_vis_path, dpi=150, bbox_inches='tight', pad_inches=0)
    print(f"Visualization saved to {output_vis_path}")
    plt.close(fig)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # Required arguments for the model
    args.add_argument("--artifact_path", type=Path, required=True, help="Path to the directory containing model checkpoints/weights.")
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