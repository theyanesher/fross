import argparse
import pickle
import os
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import numpy as np

# --- Class Definitions ---
SCANNET_CLASSES = [
    "bathtub", "bed", "bookshelf", "cabinet", "chair", "counter", "curtain", 
    "desk", "door", "floor", "otherfurniture", "picture", "refridgerator", 
    "shower curtain", "sink", "sofa", "table", "toilet", "wall", "window"
]

REPLICA_CLASSES = [
    "bag", "bskt.", "bed", "bench", "bike", "book", "botl.", "bowl", "box", 
    "cab.", "chair", "clock", "cntr.", "cup", "curt.", "desk", "door", "lamp", 
    "pil.", "plant", "plate", "pot", "rail.", "scrn.", "shlf.", "shoe", "sink", 
    "stand", "table", "toil.", "towel", "umb.", "vase", "wind."
]

SCANNET_REL_CLASSES = [
    "attached to", "build in", "connected to", "hanging on", "part of", 
    "standing on", "supported by"
]

REPLICA_REL_CLASSES = [
    "above", "against", "attached to", "has", "in", "near", "on", "under", "with"
]

np.random.seed(42)
def get_colors(n):
    return np.random.rand(n, 3)

# --- Helper to move tensors to CPU/Numpy ---
def to_numpy(x):
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

# --- Drawing Functions ---

def draw_bbox(axe, bbox, label, score, class_names, obj_colors, threshold=0.7):
    # Ensure inputs are numpy arrays
    bbox = to_numpy(bbox)
    label = to_numpy(label)
    score = to_numpy(score)
    
    # FIX 1: Filter raw scores to match the already-filtered bounding boxes
    # The pkl contains 300 scores but N bboxes. We apply the threshold to scores 
    # to get the matching N scores.
    mask = score > threshold
    
    # If lengths don't match, we assume scores corresponds to bboxes sequentially 
    # or we just take the filtered list.
    filtered_score = score[mask]
    
    # Safety check: if lengths still don't match (e.g. diff threshold used), 
    # we truncate to the shorter length to prevent crash.
    n_boxes = len(bbox)
    if len(filtered_score) != n_boxes:
        # If we can't align scores, we just use 0s or slice
        if len(filtered_score) > n_boxes:
            filtered_score = filtered_score[:n_boxes]
        else:
            # Pad if missing
            pad = np.zeros(n_boxes - len(filtered_score))
            filtered_score = np.concatenate([filtered_score, pad])
    
    score = filtered_score

    for i in range(n_boxes):
        cx, cy, w, h = bbox[i]
        x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        
        l = int(label[i])
        if l < 0 or l >= len(class_names):
            continue
            
        color = obj_colors[l]
        L = 0.2126*(color[0]**2.2) + 0.7152*(color[1]**2.2) + 0.0722*(color[2]**2.2)
        
        # Draw Box
        axe.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=2))
        
        # Draw Label
        text_label = f"{class_names[l]}: {score[i]:.2f}"
        axe.text(x1, y1, 
                 text_label, color="black" if L > 0.55 else "white", 
                 fontsize=6, fontfamily="sans-serif", ha='left', va='bottom', 
                 bbox=dict(boxstyle="square,pad=0.1",
                           facecolor=color, edgecolor=color, alpha=1))

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

def draw_rel(axe, rel, rel_label, bbox, label, rel_class_names, obj_class_names, rel_colors):
    bbox = to_numpy(bbox)
    
    # No score filtering needed for rels here, as `rel` indices usually refer to the *kept* objects
    # (assuming sg_prediction.py extracts relations on filtered objects).
    
    base_curvature = 0.2
    grouped = defaultdict(list)
    for r, l in zip(rel, rel_label):
        grouped[(r[0], r[1])].append(l)

    for (si, oi), rels in grouped.items():
        curvs = curvature_seq(len(rels), base_curvature)
        for idx, rel_idx in enumerate(rels):
            if si >= len(bbox) or oi >= len(bbox): continue
            
            sx, sy = bbox[si][0], bbox[si][1]
            ox, oy = bbox[oi][0], bbox[oi][1]
            curvature = curvs[idx]
            
            if rel_idx < 0 or rel_idx >= len(rel_class_names): continue
            rel_color = rel_colors[rel_idx]

            arrow = FancyArrowPatch(
                (sx, sy), (ox, oy),
                connectionstyle=f"arc3,rad={curvature}",
                arrowstyle='-|>',
                mutation_scale=10,
                linewidth=1,
                color=rel_color,
                alpha=0.8
            )
            axe.add_patch(arrow)
            lx, ly = arc3_midpoint((sx, sy), (ox, oy), curvature)

            L = 0.2126*(rel_color[0]**2.2) + 0.7152*(rel_color[1]**2.2) + 0.0722*(rel_color[2]**2.2)
            axe.text(
                lx, ly,
                rel_class_names[rel_idx],
                ha="center", va="center",
                fontsize=5, color="black" if L > 0.55 else "white", fontfamily="sans-serif",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=rel_color, edgecolor=rel_color, alpha=0.9)
            )

def main(args):
    scene = args.scene
    dataset_path = f"{args.dataset_path}/{scene}/sequence/"
    dataset = "ReplicaSSG" if "Replica" in dataset_path else "3RScan"
    
    if dataset == "ReplicaSSG":
        CLASSES = REPLICA_CLASSES
        REL_CLASSES = REPLICA_REL_CLASSES
    else:
        CLASSES = SCANNET_CLASSES
        REL_CLASSES = SCANNET_REL_CLASSES
        
    OBJ_COLORS = get_colors(len(CLASSES))
    REL_COLORS = get_colors(len(REL_CLASSES))

    print(f"Visualizing 2D SG for scene: {scene} ({dataset})")

    with open(f"{args.vis_folder}/{dataset}/{scene}/obj_2d.pkl", "rb") as f:
        obj_2d = pickle.load(f)
    with open(f"{args.vis_folder}/{dataset}/{scene}/rel_2d.pkl", "rb") as f:
        rel_2d = pickle.load(f)

    # FIX 2: Exclude 'rendered' images to prevent duplicates
    imgs = sorted([
        img for img in os.listdir(dataset_path) 
        if img.endswith('.color.jpg') and 'rendered' not in img
    ])
    
    os.makedirs(f"{args.vis_folder}/2D/{scene}", exist_ok=True)

    for idx, img in enumerate(tqdm(imgs, desc=f"{scene}: ")):
        I = Image.open(os.path.join(dataset_path, img))
        
        if dataset == "3RScan":
            I = I.transpose(Image.ROTATE_270)

        if idx >= len(obj_2d): break

        bboxes = obj_2d[idx]
        rels_data = rel_2d[idx]
        
        obj_classes = to_numpy(bboxes['classes'])
        obj_boxes = to_numpy(bboxes['bboxes'])
        obj_scores = to_numpy(bboxes['scores'])
        
        rel_classes_ids = rels_data['rel_classes']
        rels_edges = rels_data['rels']
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(I)
        
        draw_rel(ax, rels_edges, rel_classes_ids, obj_boxes, obj_classes, REL_CLASSES, CLASSES, REL_COLORS)
        # Pass the threshold to align scores
        draw_bbox(ax, obj_boxes, obj_classes, obj_scores, CLASSES, OBJ_COLORS, threshold=args.obj_thresh)
        
        ax.axis('off')
        fig.set_size_inches(I.size[0] / 300, I.size[1] / 300) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{args.vis_folder}/2D/{scene}/{img[:-4]}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--scene", type=str, required=True)
    args.add_argument("--vis_folder", type=str, required=True)
    # Added argument to match the threshold used in inference
    args.add_argument("--obj_thresh", type=float, default=0.7, help="Object threshold used during inference to align scores.")
    args = args.parse_args()
    main(args)