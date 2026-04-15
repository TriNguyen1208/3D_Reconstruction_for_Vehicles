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

    def update_pose_metrics(self, pred_extrinsics, gt_poses):
        """
        This function processes and stores camera pose errors for a sequence of frames.

        Args:
            pred_extrinsics (list or np.ndarray): Extrinsic camera matrices for all frames 
                (e.g., shape N x 3 x 4).
            gt_poses (list or np.ndarray): Ground Truth ARFrame matrices for all frames 
                (e.g., shape N x 16).

        Returns:
            tuple: (batch_rra, batch_rta) containing lists of errors for the processed sequence.
        """
        
        batch_rra = []
        batch_rta = []
        
        for p_ext, g_pose in zip(pred_extrinsics, gt_poses):
            R_p = p_ext[:3, :3]
            t_p = p_ext[:3, 3]
            
            R_g, t_g = self.convert_ar_pose_to_opencv(g_pose)
            
            rra = self.compute_rra(R_p, R_g)
            rta = self.compute_rta(t_p, t_g)
            
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
        
        result = {
            "num_frames": len(self.accuracies),
            "auc": auc_score,
            "acc_mean": np.mean(self.accuracies) if self.accuracies else float('inf'),
            "comp_mean": np.mean(self.completenesses) if self.completenesses else float('inf'),
            "overall_cd": (np.mean(self.accuracies) + np.mean(self.completenesses)) / 2 if self.accuracies else float('inf')
        }
        
        print(result)
        
        return result



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
    
    def get_gt_poses(folder_path):
        json_files = sorted(glob.glob(os.path.join(folder_path, "frame_*.json")))
        gt_trajectories = []
        
        for f in json_files:
            with open(f, 'r') as j:
                data = json.load(j)
                matrix = np.array(data['cameraPoseARFrame']).reshape(4, 4)
                translation = matrix[:3, 3]
                gt_trajectories.append(translation)
                
        return np.array(gt_trajectories)