from vggt.utils.load_fn import load_and_preprocess_images
import glob
import os
import torch
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import numpy as np

def run_model(target_dir, model, yolo, device, is_fg_mask=True) -> dict:
    """
    Run VGGT on frame_*.jpg images directly in target_dir.
    Model should already be on device and in eval mode before calling this.
    """
    print(f"Processing images from {target_dir}")

    image_names = sorted(glob.glob(os.path.join(target_dir, "*.jpg")))
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError(f"No images found in {target_dir}.")

    images = load_and_preprocess_images(image_names).to(device)

    if yolo is not None and is_fg_mask:
        fg_mask, images = yolo.process_and_blend_images(images)
    else:
        fg_mask = None
    
    vggt_h, vggt_w = images.shape[-2], images.shape[-1]
    print(f"Preprocessed images shape: {images.shape}")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Store VGGT resolution for focal rescaling in evaluation
    predictions["vggt_resolution"] = (vggt_h, vggt_w)

    # Convert tensors → numpy, remove batch dim
    skip_keys = {"pose_enc_list"}

    for key in list(predictions.keys()):
        print(key)
        if key in skip_keys:
            predictions[key] = None
            continue
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # World points from depth
    depth_map = predictions["depth"]  # [S, H, W, 1]
    
    world_points = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )

    if yolo is not None and fg_mask is not None:
        mask_expanded = fg_mask.unsqueeze(-1).cpu().numpy().astype(np.float32)
        mask_np = fg_mask.cpu().numpy().astype(np.float32)
        
        predictions["world_points_from_depth"] = world_points * mask_expanded
        
        if "world_points" in predictions:
            predictions["world_points"] = predictions["world_points"] * mask_expanded
        if "world_points_conf" in predictions:
            predictions["world_points_conf"] = predictions["world_points_conf"] * mask_np
    else:
        predictions["world_points_from_depth"] = world_points
    predictions["images"] = images.cpu().numpy()
    torch.cuda.empty_cache()
    
    return predictions