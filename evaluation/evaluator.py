import numpy as np
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

    def update_pose_metrics(self, pred_extrinsic, gt_pose):
        """
        This function processes and stores camera pose errors for a single frame.

        Args:
            pred_extrinsic (np.ndarray): A 3x4 extrinsic camera matrix 
                from the model prediction.
            gt_pose (list): A 16-item flat list representing the 4x4 transformation 
                matrix (Column-Major) from the Ground Truth ARFrame.

        Returns:
            tuple: A tuple (rra, rta) containing:
                - rra (float): Relative Rotation Accuracy in degrees.
                - rta (float): Relative Translation Accuracy (angular) in degrees.
        """
        
        R_p = pred_extrinsic[:3, :3]
        t_p = pred_extrinsic[:3, 3]
        
        R_g, t_g = self.convert_ar_pose_to_opencv(gt_pose)
        
        rra = self.compute_rra(R_p, R_g)
        rta = self.compute_rta(t_p, t_g)
        
        self.rra_errors.append(rra)
        self.rta_errors.append(rta)
        
        return rra, rta

    def update_geometric_metrics(self, pts_pred_aligned, pts_gt_raw, car_mask_pred):
        """
        This function computes and stores semantic-aware Chamfer Distance metrics (Accuracy and Completeness) for a single frame.

        Args:
            pts_pred_aligned (np.ndarray): Aligned predicted point cloud (N, 3).
            pts_gt_raw (np.ndarray): Raw Ground Truth point cloud (M, 3).
            car_mask_pred (np.ndarray): Binary segmentation mask for the target vehicle.

        Returns:
            tuple: (accuracy, completeness, overall) scores for the current frame.
        """
        
        acc, comp, overall = self.compute_chamfer_distance(
            pts_pred_aligned, 
            pts_gt_raw, 
            car_mask_pred, 
            self.num_samples
        )
        
        if acc != float('inf') and comp != float('inf'):
            self.accuracies.append(acc)
            self.completenesses.append(comp)
            
        return acc, comp, overall

    def get_summary(self):
        """
        This function aggregates stored frame data to compute final benchmark metrics including AUC@30 and Mean Chamfer Distance.

        Args:
            None (uses internal state from accumulated frames).

        Returns:
            dict: A dictionary containing the sequence-level AUC score, mean Accuracy, mean Completeness, and Overall CD.
        """
        
        auc_score = self.compute_pose_auc(
            self.rra_errors, 
            self.rta_errors, 
            self.auc_threshold
        )
        
        return {
            "num_frames": len(self.accuracies),
            "auc": auc_score,
            "acc_mean": np.mean(self.accuracies) if self.accuracies else float('inf'),
            "comp_mean": np.mean(self.completenesses) if self.completenesses else float('inf'),
            "overall_cd": (np.mean(self.accuracies) + np.mean(self.completenesses)) / 2 if self.accuracies else float('inf')
        }



    @staticmethod
    def apply_umeyama_alignment(source, target):
        """
        This function computes the optimal similarity transform (rotation, translation, and scale) to align two corresponding point sets.

        Args:
            source (np.ndarray): Predicted points of shape (N, 3).
            target (np.ndarray): Ground Truth points of shape (N, 3).

        Returns:
            tuple: (R, t, s, transformed_source) containing the 3x3 rotation matrix, 3x1 translation vector, scale factor, and the aligned source points.
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
        
        transformed_source = s * (R @ source.T).T + t
        return R, t, s, transformed_source

    @staticmethod
    def convert_ar_pose_to_opencv(gt_pose):
        """
        This function converts a 16-element OpenGL-style column-major matrix to OpenCV-style rotation and translation.

        Args:
            gt_pose (list/np.ndarray): A 16-element flat list representing the 4x4 transformation matrix.

        Returns:
            tuple: (R_opencv, t_opencv) representing the 3x3 rotation matrix and 3x1 translation vector in OpenCV coordinates.
        """
        
        gt_matrix = np.array(gt_pose).reshape(4, 4).T
        R_raw = gt_matrix[:3, :3]
        t_raw = gt_matrix[:3, 3]

        flip_yz = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        
        R_opencv = np.dot(flip_yz, R_raw)
        t_opencv = np.dot(flip_yz, t_raw)
        return R_opencv, t_opencv

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
        
        auc_val = np.trapezoid(recalls, thresholds)
        return (auc_val / threshold) * 100

    @staticmethod
    def compute_chamfer_distance(pts_pred, pts_gt, car_mask_pred, num_samples=5000):
        """
        This function computes the Accuracy, Completeness, and Overall Chamfer Distance using masked and sampled point clouds.

        Args:
            pts_pred (np.ndarray): Predicted point cloud (N, 3).
            pts_gt (np.ndarray): Ground truth point cloud (M, 3).
            car_mask_pred (np.ndarray): Semantic mask for filtering predicted car points.
            num_samples (int): Number of points to sample for distance computation (default is 5000).

        Returns:
            tuple: (accuracy, completeness, overall_cd) scores.
        """
        
        clean_pred = pts_pred[car_mask_pred.squeeze() > 0.5]
        
        if len(clean_pred) > 0:
            min_b, max_b = clean_pred.min(axis=0), clean_pred.max(axis=0)
            padding = 0.1
            gt_mask = np.all((pts_gt >= min_b - padding) & (pts_gt <= max_b + padding), axis=1)
            clean_gt = pts_gt[gt_mask]
        else:
            return float('inf'), float('inf'), float('inf')

        if len(clean_pred) == 0 or len(clean_gt) == 0:
            return float('inf'), float('inf'), float('inf')

        def get_sample(points, n):
            if len(points) > n:
                return points[np.random.choice(len(points), n, replace=False)]
            return points

        s_pred = get_sample(clean_pred, num_samples)
        s_gt = get_sample(clean_gt, num_samples)

        tree_gt = cKDTree(s_gt)
        dist_p, _ = tree_gt.query(s_pred)
        accuracy = np.mean(np.square(dist_p))
        
        tree_pred = cKDTree(s_pred)
        dist_g, _ = tree_pred.query(s_gt)
        completeness = np.mean(np.square(dist_g))

        return accuracy, completeness, (accuracy + completeness) / 2.0