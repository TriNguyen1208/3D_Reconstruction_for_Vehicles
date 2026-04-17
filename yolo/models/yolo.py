from ultralytics import YOLO
import cv2
import torch
import numpy as np

class YoloSegment:
    def __init__(self):
        self.vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]
        self.model = None

    def load_model(self, model_name="yolov8s-seg.pt"):
        self.model = YOLO(model_name)

    def process_and_blend_images(self, images_tensor, bg_blur_ksize=(51, 51), mask_blur_ksize=(15, 15)):
        """
        Thực hiện YOLO Segmentation và Alpha Blending trực tiếp trên Tensor.
        Trả về:
            - fg_masks: Tensor nhị phân [S, H, W] dùng để lọc Point Cloud.
            - blended_images: Tensor ảnh [S, C, H, W] đã làm mờ nền dùng cho VGGT.
        """
        # 1. Lưu lại thông tin Device và Dtype gốc để trả về đúng định dạng
        orig_device = images_tensor.device
        orig_dtype = images_tensor.dtype

        images = images_tensor.clone()
        is_4d = False
        if images.dim() == 4:
            is_4d = True
            images = images.unsqueeze(0)  # [1, S, C, H, W]

        B, S, C, H, W = images.shape
        images = images.view(B * S, C, H, W)
        
        fg_masks = []
        blended_images = []

        for img_t in images:
            # 2. Chuyển Tensor -> Numpy mảng Float [0, 1]
            img_np = img_t.detach().cpu().permute(1, 2, 0).numpy()
            
            # Xử lý lỗi ảnh rỗng
            if img_np.size == 0 or img_np.shape[0] == 0 or img_np.shape[1] == 0:
                print("Cảnh báo: Ảnh không hợp lệ, bỏ qua xử lý YOLO.")
                fg_masks.append(torch.zeros((H, W), dtype=orig_dtype))
                blended_images.append(img_t)
                continue

            # 3. Chuẩn bị ảnh uint8 cho YOLO
            img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            img_bgr = np.ascontiguousarray(img_bgr)

            # 4. Chạy YOLO
            pred_results = self.model.predict(
                source=[img_bgr], conf=0.5, task="segment", save=False, verbose=False
            )
    
            best_mask = None
            if pred_results and pred_results[0].masks is not None:
                result = pred_results[0]
                img_h, img_w = result.orig_shape
                center_x_img = img_w / 2
                min_distance = float('inf')
                masks_data = result.masks.data
                best_box = None
                # Tìm xe gần tâm ảnh nhất
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0].item())
                    class_name = self.model.names[class_id]
                    if class_name not in self.vehicle_classes:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_center_x = (x1 + x2) / 2
                    distance = abs(box_center_x - center_x_img)

                    if distance < min_distance:
                        min_distance = distance
                        best_mask = masks_data[i]
                        best_box = box

            # 5. Xử lý Blending nếu tìm thấy xe
            # if best_mask is not None:
            #     # Mask thô [H, W]
            #     mask_np = best_mask.detach().cpu().numpy()
            #     if mask_np.shape != (H, W):
            #         mask_np = cv2.resize(
            #             mask_np.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
            #         )
                
            #     # A. Binary Mask (Để nguyên 0 và 1 phục vụ lọc Point Cloud sau này)
            #     binary_mask = (mask_np > 0.5).astype(np.float32)
                
            #     # B. Soft Mask (Làm mờ viền mask để ghép ảnh không bị răng cưa)
            #     soft_mask = cv2.GaussianBlur(binary_mask, mask_blur_ksize, 0)
            #     soft_mask_3d = soft_mask[:, :, None]  # [H, W, 1]
                
            #     # C. Làm mờ ảnh nền (Gaussian Blur)
            #     bg_blurred = cv2.GaussianBlur(img_np, bg_blur_ksize, 0)
                
            #     # D. ALPHA BLENDING: Giữ nguyên vùng xe, làm mờ vùng nền
            #     # Thực hiện trên hệ Float [0, 1] để tránh sai số
            #     blended_np = img_np * soft_mask_3d + bg_blurred * (1.0 - soft_mask_3d)
                
            #     # Trả về Tensor
            #     blended_t = torch.from_numpy(blended_np).permute(2, 0, 1).to(orig_dtype)
            #     mask_t = torch.from_numpy(binary_mask).to(orig_dtype)
                
            #     blended_images.append(blended_t)
            #     fg_masks.append(mask_t)
            # else:
            #     # Fallback: Không tìm thấy xe thì giữ nguyên ảnh gốc và mask = 0
            #     fg_masks.append(torch.zeros((H, W), dtype=orig_dtype))
            #     blended_images.append(img_t)

            #6: Test với box
            if best_box is not None:
                binary_mask = np.zeros((H, W), dtype=np.float32)
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                binary_mask[y1:y2, x1:x2] = 1.0
                
                # B. Soft Mask (Làm mờ viền mask để ghép ảnh không bị răng cưa)
                soft_mask = cv2.GaussianBlur(binary_mask, mask_blur_ksize, 0)
                soft_mask_3d = soft_mask[:, :, None]  # [H, W, 1]
                
                # C. Làm mờ ảnh nền (Gaussian Blur)
                bg_blurred = cv2.GaussianBlur(img_np, bg_blur_ksize, 0)
                
                # D. ALPHA BLENDING: Giữ nguyên vùng xe, làm mờ vùng nền
                # Thực hiện trên hệ Float [0, 1] để tránh sai số
                blended_np = img_np * soft_mask_3d + bg_blurred * (1.0 - soft_mask_3d)
                
                # Trả về Tensor
                blended_t = torch.from_numpy(blended_np).permute(2, 0, 1).to(orig_dtype)
                mask_t = torch.from_numpy(binary_mask).to(orig_dtype)
                
                blended_images.append(blended_t)
                fg_masks.append(mask_t)
            else:
                # Fallback: Không tìm thấy xe thì giữ nguyên ảnh gốc và mask = 0
                fg_masks.append(torch.zeros((H, W), dtype=orig_dtype))
                blended_images.append(img_t)

        # 6. Gom lại thành Batch và đưa về Device gốc (GPU nếu có)
        final_masks = torch.stack(fg_masks, dim=0).to(orig_device)
        final_blended = torch.stack(blended_images, dim=0).to(orig_device)

        if is_4d:
            # Trả về đúng shape ban đầu [S, C, H, W] và [S, H, W]
            pass
        else:
            final_masks = final_masks.view(B, S, H, W)
            final_blended = final_blended.view(B, S, C, H, W)

        return final_masks, final_blended