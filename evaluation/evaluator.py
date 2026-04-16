import numpy as np
import os
import json
import trimesh
import glob
from scipy.spatial import cKDTree

class Evaluator:
    def __init__(self, auc_threshold=30, num_samples=5000):
        self.auc_threshold = auc_threshold
        self.num_samples = num_samples
        
        # State storage for sequence evaluation
        self.rra_errors = []
        self.rta_errors = []
        self.accuracies = []
        self.completenesses = []

    def update_pose_metrics(self, pred_extrinsics, gt_poses):
        """
        This function processes and stores camera pose errors for a sequence of frames.

        Args:
            pred_extrinsics (list or np.ndarray): Extrinsic camera matrices for all frames 
                (e.g., shape N x 3 x 4).
            gt_poses (list or np.ndarray): Ground Truth ARFrame matrices for all frames 
                (e.g., shape N x 4 x 4).

        Returns:
            tuple: (batch_rra, batch_rta) containing lists of errors for the processed sequence.
        """
        
        batch_rra = []
        batch_rta = []
        
        standard_pred = self.get_relative_pose(pred_extrinsics)
        standard_gt_poses = self.get_relative_pose(gt_poses)
        
        pred_xyz = np.array([p[:3, 3] for p in standard_pred])
        gt_xyz = np.array([g[:3, 3] for g in standard_gt_poses])
        
        R_u, t_u, s_u = self.apply_umeyama_alignment(pred_xyz, gt_xyz)
        
        for i in range(len(standard_pred)):
            R_p = standard_pred[i][:3, :3]
            R_g = standard_gt_poses[i][:3, :3]
            rra = self.compute_rra(R_p, R_g)
            
            t_p_original = standard_pred[i][:3, 3]
            aligned_t_p = s_u * (R_u @ t_p_original) + t_u
            
            t_g = standard_gt_poses[i][:3, 3]
            rta = self.compute_rta(aligned_t_p, t_g)
            
            self.rra_errors.append(rra)
            self.rta_errors.append(rta)
            batch_rra.append(rra)
            batch_rta.append(rta)
            
        return batch_rra, batch_rta

    def update_geometric_metrics(self, pts_preds_raw, gt_vertices):
        """
        Modified to compare frame-wise predictions against a static global mesh.
        """
        batch_acc, batch_comp, batch_overall = [], [], []
        
        for p_pred in pts_preds_raw:
            # 1. Alignment
            # Since there is no pixel-correspondence with a mesh, 
            # we assume points are either already in world space 
            # or we align using the centroids/bboxes.
            p_pred_flat = p_pred.reshape(-1, 3)
            
            # 2. Compute Chamfer against the WHOLE mesh
            acc, comp, overall = self.compute_chamfer_distance(
                p_pred_flat, 
                gt_vertices, # Compare against the full static mesh
                self.num_samples
            )
            
            if acc != float('inf'):
                self.accuracies.append(acc)
                self.completenesses.append(comp)
                
            batch_acc.append(acc)
            batch_comp.append(comp)
            batch_overall.append(overall)
            
        return batch_acc, batch_comp, batch_overall

    def get_summary(self):
        """
        Aggregates frame data, computes final metrics, and prints a formatted 
        summary table for research reporting.
        """
        auc_score = self.compute_pose_auc(
            self.rra_errors, 
            self.rta_errors, 
            self.auc_threshold
        )
        
        # Calculate means
        acc_mean = np.mean(self.accuracies) if self.accuracies else float('inf')
        comp_mean = np.mean(self.completenesses) if self.completenesses else float('inf')
        overall_cd = (acc_mean + comp_mean) / 2 if self.accuracies else float('inf')
        
        # Helpful for debugging your 0.0 AUC
        avg_rra = np.mean(self.rra_errors) if self.rra_errors else 0.0
        avg_rta = np.mean(self.rta_errors) if self.rta_errors else 0.0

        result = {
            "num_frames": len(self.accuracies),
            "auc": auc_score,
            "acc_mean": acc_mean,
            "comp_mean": comp_mean,
            "overall_cd": overall_cd,
            "rra_avg": avg_rra,
            "rta_avg": avg_rta
        }

        # Professional Table Printing
        print("\n" + "="*45)
        print(f"{'3D RECONSTRUCTION EVALUATION SUMMARY':^45}")
        print("="*45)
        print(f"{'Metric':<25} | {'Value':<15}")
        print("-" * 45)
        print(f"{'Frames Processed':<25} | {result['num_frames']:<15}")
        print(f"{'AUC@' + str(self.auc_threshold):<25} | {result['auc']:<15.2f}")
        print(f"{'Avg RRA (Rotation °)':<25} | {result['rra_avg']:<15.2f}")
        print(f"{'Avg RTA (Transl. °)':<25} | {result['rta_avg']:<15.2f}")
        print("-" * 45)
        print(f"{'Accuracy':<25} | {result['acc_mean']:<15.6f}")
        print(f"{'Completeness':<25} | {result['comp_mean']:<15.6f}")
        print(f"{'Overall CD':<25} | {result['overall_cd']:<15.6f}")
        print("="*45 + "\n")

        return result

    @staticmethod
    def get_relative_pose(poses):
        poses = np.array(poses) 
        
        if poses.shape[-2] == 3: 
            new_poses = []
            for p in poses:
                new_poses.append(np.vstack([p, [0, 0, 0, 1]]))
            poses = np.array(new_poses)

        inv_p0 = np.linalg.inv(poses[0])
        
        relative_poses = []
        for pi in poses:
            relative_poses.append(inv_p0 @ pi)
            
        return np.array(relative_poses)


    @staticmethod
    def apply_umeyama_alignment(source, target):
        """
        Computes the optimal similarity transform (R, t, s) to align source to target.
        """
        n, m = source.shape
        mu_s = source.mean(axis=0)
        mu_t = target.mean(axis=0)
        
        s_centered = source - mu_s
        t_centered = target - mu_t
        
        var_s = np.mean(np.sum(s_centered**2, axis=1))
        cov = (t_centered.T @ s_centered) / n
        
        u, d, vh = np.linalg.svd(cov)
        
        S = np.eye(m)
        if np.linalg.det(u) * np.linalg.det(vh) < 0:
            S[m-1, m-1] = -1
            
        R = u @ S @ vh
        s = (1.0 / var_s) * np.trace(np.diag(d) @ S)
        t = mu_t - s * (R @ mu_s)
        
        # transformed_source = s * (R @ source.T).T + t
        return R, t, s

    @staticmethod
    def convert_ar_pose_to_opencv(gt_pose):
        if isinstance(gt_pose, list) or (isinstance(gt_pose, np.ndarray) and gt_pose.size == 16):
            gt_matrix = np.array(gt_pose).reshape(4, 4)
        else:
            gt_matrix = gt_pose
            
        R_raw = gt_matrix[:3, :3]
        t_raw = gt_matrix[:3, 3]
        
        transform = np.array([
            [1,  0,  0],
            [ 0,  -1,  0],
            [ 0, 0,  -1]
        ])
        
        R_opencv = transform @ R_raw
        t_opencv = transform @ t_raw
        
        # return R_opencv, t_opencv
        return R_raw, t_raw

    @staticmethod
    def compute_rra(R_pred, R_gt):
        """
        This function calculates the Relative Rotation Accuracy (RRA) between predicted and ground truth rotation matrices.

        Args:
            R_pred (np.ndarray): Predicted 3x3 rotation matrix.
            R_gt (np.ndarray): Ground truth 3x3 rotation matrix.

        Returns:
            float: The rotation error in degrees.
        """
        
        R_rel = np.dot(R_pred, R_gt.T)
        trace = np.trace(R_rel)
        cos_theta = (trace - 1.0) / 2.0
        error_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(error_rad)

    @staticmethod
    def compute_rta(t_pred, t_gt):
        """
        This function calculates the Relative Translation Accuracy (RTA) as the angular error between translation vectors.

        Args:
            t_pred (np.ndarray): Predicted 3x1 translation vector.
            t_gt (np.ndarray): Ground truth 3x1 translation vector.

        Returns:
            float: The angular translation error in degrees.
        """
        
        unit_t_p = t_pred / (np.linalg.norm(t_pred) + 1e-8)
        unit_t_g = t_gt / (np.linalg.norm(t_gt) + 1e-8)
        
        cos_theta = np.dot(unit_t_p, unit_t_g)
        error_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(error_rad)

    @staticmethod
    def compute_pose_auc(rra_list, rta_list, threshold=30):
        """
        This function computes the Area Under Curve (AUC) for the combined pose error at a specified threshold.

        Args:
            rra_list (list): List of rotation errors in degrees.
            rta_list (list): List of translation errors in degrees.
            threshold (int): The maximum error threshold for AUC calculation (default is 30).

        Returns:
            float: The AUC score normalized to a 0-100 scale.
        """
        
        if not rra_list or not rta_list: return 0.0
        
        rra = np.array(rra_list)
        rta = np.array(rta_list)
        
        combined_errors = np.maximum(rra, rta)
        num_frames = len(combined_errors)
        
        thresholds = np.linspace(0, threshold, 100)
        recalls = np.sum(combined_errors[:, None] <= thresholds, axis=0) / num_frames
        
        auc_val = np.trapz(recalls, thresholds)
        return (auc_val / threshold) * 100

    @staticmethod
    def compute_chamfer_distance(pts_pred, pts_gt, num_samples=5000):
        """
        Computes Accuracy, Completeness, and Overall CD after cleaning zero points.
        """
        clean_pred = pts_pred[np.any(pts_pred != 0, axis=-1)]
        clean_gt = pts_gt[np.any(pts_gt != 0, axis=-1)]

        if len(clean_pred) == 0 or len(clean_gt) == 0:
            return float('inf'), float('inf'), float('inf')

        def get_sample(points, n):
            if len(points) > n:
                return points[np.random.choice(len(points), n, replace=False)]
            return points

        s_pred = get_sample(clean_pred, num_samples)
        s_gt = get_sample(clean_gt, num_samples)

        tree_gt = cKDTree(s_gt)
        accuracy = np.mean(np.square(tree_gt.query(s_pred)[0]))
        
        tree_pred = cKDTree(s_pred)
        completeness = np.mean(np.square(tree_pred.query(s_gt)[0]))

        return accuracy, completeness, (accuracy + completeness) / 2.0
    
    @staticmethod
    def get_gt_points(folder_path):
        """
        This function searches for a single .obj file in the specified folder 
        and returns its vertices as a point cloud.

        Args:
            folder_path (str): The path to the folder containing the Ground Truth mesh.

        Returns:
            np.ndarray: The vertices of the mesh of shape (M, 3).

        Raises:
            FileNotFoundError: If no .obj file is found in the folder.
            ValueError: If more than one .obj file is found.
        """
        obj_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.obj')]

        if len(obj_files) == 0:
            raise FileNotFoundError(f"Error: No .obj files found in {folder_path}")
        
        if len(obj_files) > 1:
            raise ValueError(
                f"Error: Multiple .obj files found in {folder_path}. "
                f"Found: {obj_files}. Please ensure only one GT mesh is present."
            )

        mesh_path = os.path.join(folder_path, obj_files[0])
        gt_mesh = trimesh.load(mesh_path)
        
        gt_points = np.array(gt_mesh.vertices)
        
        return gt_points
    
    @staticmethod
    def get_gt_poses(folder_path):
        json_files = sorted(glob.glob(os.path.join(folder_path, "frame_*.json")))
        gt_matrices = []

        for f in json_files:
            with open(f, 'r') as j:
                data = json.load(j)
                matrix = np.array(data['cameraPoseARFrame']).reshape(4, 4)
                gt_matrices.append(matrix)
                    
        return gt_matrices
