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
        gt_poses = evaluator.get_gt_poses(args.path_dataset)
        
        import numpy as np
        
        def get_relative_pose(poses):
            # Đảm bảo poses là numpy array
            poses = np.array(poses) 
            
            # Kiểm tra nếu là (S, 3, 4), ta cần chuyển tất cả sang (S, 4, 4)
            if poses.shape[-2] == 3: 
                new_poses = []
                for p in poses:
                    new_poses.append(np.vstack([p, [0, 0, 0, 1]]))
                poses = np.array(new_poses)

            # Tính ma trận nghịch đảo của khung hình đầu tiên
            inv_p0 = np.linalg.inv(poses[0])
            
            relative_poses = []
            for pi in poses:
                # Phép nhân ma trận đồng nhất: P0^-1 * Pi
                relative_poses.append(inv_p0 @ pi)
                
            return np.array(relative_poses)

        standard_gt_poses = get_relative_pose(gt_poses)
        standard_pred = get_relative_pose(predictions["extrinsic"])
        
        gt_xyz = standard_gt_poses[:, :3, 3]
        pred_xyz = standard_pred[:, :3, 3]

        # 3. Tìm bộ tham số nắn quỹ đạo (R, t, s)
        R_u, t_u, s_u = evaluator.apply_umeyama_alignment(pred_xyz, gt_xyz)

        # 4. Tạo ma trận PRED đã được căn chỉnh (Aligned) để in ra so sánh
        aligned_standard_pred = []
        for i in range(len(standard_pred)):
            p = standard_pred[i]
            # Biến đổi phần Rotation: R_aligned = R_u @ R_original
            new_R = R_u @ p[:3, :3]
            # Biến đổi phần Translation: T_aligned = s * (R_u @ T_original) + t_u
            new_t = s_u * (R_u @ p[:3, 3]) + t_u
            
            # Ghép lại thành ma trận 4x4
            aligned_p = np.eye(4)
            aligned_p[:3, :3] = new_R
            aligned_p[:3, 3] = new_t
            aligned_standard_pred.append(aligned_p)

        # 5. In kết quả so sánh
        for i in range(len(standard_gt_poses)):
            print(f'\nFRAME {i}')
            print('=' * 20 + ' GT (Relative) ' + '=' * 20)
            print(np.round(standard_gt_poses[i], 5))
            print('=' * 20 + ' PRED (Aligned & Scaled) ' + '=' * 20)
            print(np.round(aligned_standard_pred[i], 5))

        print(f"\n---> Hệ số tỷ lệ (Scale Factor) tìm được: {s_u:.4f}")
        # print("Full Matrix:\n", predictions['extrinsic'][0, 0].cpu().numpy())
        # evaluator = Evaluator()

        # gt_points = evaluator.get_gt_points(args.path_dataset)        
        # gt_poses = evaluator.get_gt_poses(args.path_dataset)
    
        # evaluator.update_pose_metrics(predictions["extrinsic"], gt_poses)
        # evaluator.update_geometric_metrics(predictions["world_points"], gt_points)
        # evaluator.get_summary()

if __name__ == "__main__":
    main()