import argparse
import os
import trimesh
import torch

from vggt.models.vggt import VGGT
from yolo.models.yolo import YoloSegment
from utils.run_model import run_model
from utils.visual_util import save_to_obj
from evaluation.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Reconstruction Pipeline CLI"
    )

    # Required
    parser.add_argument(
        "-p", "--path-dataset", type=str, required=True,
        help="Path to dataset folder containing multi-view images"
    )

    # Flags
    parser.add_argument(
        "-fg", "--fg-mask", action="store_true",
        help="Enable foreground mask"
    )

    parser.add_argument(
        "-m", "--metrics", action="store_true",
        help="Enable metrics (requires dataset contains .json and groundtruth.obj)"
    )

    parser.add_argument(
        "-o", "--export-obj", action="store_true",
        help="Export OBJ file"
    )

    # Optional paths
    parser.add_argument(
        "-mp", "--metrics-path", type=str,
        help="Path to metrics file (default: results.txt)"
    )

    parser.add_argument(
        "-op", "--obj-path", type=str,
        help="Path to output OBJ file (default: result.obj)"
    )

    args = parser.parse_args()

    # ===================== VALIDATION =====================

    # Check dataset path tồn tại
    if not os.path.exists(args.path_dataset):
        parser.error(f"Dataset path does not exist: {args.path_dataset}")

    # Sai logic: có path nhưng không bật flag
    if args.obj_path and not args.export_obj:
        parser.error("--obj-path requires --export-obj")

    if args.metrics_path and not args.metrics:
        parser.error("--metrics-path requires --metrics")

    # ===================== DEFAULT LOGIC =====================

    if args.metrics:
        args.metrics_path = args.metrics_path or "results.txt"
    else:
        args.metrics_path = None

    if args.export_obj:
        args.obj_path = args.obj_path or "result.obj"
    else:
        args.obj_path = None

    return args


def main():
    args = parse_args()

    if "model" not in globals():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    
    if "yolo" not in globals():
        yolo = YoloSegment()
        yolo.load_model()

    predictions = run_model(target_dir=args.path_dataset, model=model, yolo=yolo, device=device, is_fg_mask=args.fg_mask)

    if args.export_obj:
        save_to_obj(predictions=predictions, obj_path=args.obj_path, is_fg_mask=args.fg_mask)

    if args.metrics:
        evaluator = Evaluator()

        gt_points = evaluator.get_gt_points(args.path_dataset)        
        gt_poses = evaluator.get_gt_poses(args.path_dataset)
    
        evaluator.update_pose_metrics(predictions["extrinsic"], gt_poses)
        evaluator.update_geometric_metrics(predictions["world_points"], gt_points)
        evaluator.get_summary()

if __name__ == "__main__":
    main()