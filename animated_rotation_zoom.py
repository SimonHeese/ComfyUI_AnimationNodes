import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math
import json

class AnimatedRotationZoom:
    MAX_RESOLUTION = 16384
    
    BEZIER_POINTS = {
        "linear": {"p1": (0.4, 0.4), "p2": (0.6, 0.6)},
        "fast":   {"p1": (0.1, 0.8), "p2": (0.9, 0.2)},
        "slow":   {"p1": (0.8, 0.1), "p2": (0.2, 0.9)},
    }

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "angle_start": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1}),
            "angle_end": ("FLOAT", {"default": 45.0, "min": -3600.0, "max": 3600.0, "step": 0.1}),
            "scale_start": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            "scale_end": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            "center_x": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01}),
            "center_y": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01}),
            "feathering": ("INT", {"default": 40, "min": 0, "max": s.MAX_RESOLUTION, "step": 1}),
            "sampling_mode": (["bilinear", "nearest", "bicubic"],),
            "anim_start": (["linear", "fast", "slow"],),
            "anim_end": (["linear", "fast", "slow"],),
        },
        "optional": {
            "coordinates": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK", "ATTN_MASK")
    FUNCTION = "rotate_and_zoom"
    CATEGORY = "image/transform"

    def parse_coordinates(self, coordinates_str):
        try:
            coords = json.loads(coordinates_str.replace("'", '"'))
            return [(float(coord['x']), float(coord['y'])) for coord in coords]
        except:
            print("Warning: Could not parse coordinates string")
            return None

    def get_center_for_frame(self, frame_idx, coordinates, default_x, default_y, W, H):
        if coordinates is None:
            return [default_x * W, default_y * H]
        try:
            return coordinates[frame_idx]
        except IndexError:
            return [default_x * W, default_y * H]

    def bezier(self, t, start_style, end_style):
        p1 = self.BEZIER_POINTS[start_style]["p1"]
        p2 = self.BEZIER_POINTS[end_style]["p2"]
        
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        
        return mt3 * 0 + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * 1

    def create_base_mask(self, h, w, feathering, device="cpu"):
        mask = torch.ones((1, h, w), device=device)
        
        if feathering > 0:
            y = torch.linspace(0, h-1, h, device=device)
            x = torch.linspace(0, w-1, w, device=device)
            
            top = y.view(-1, 1)
            left = x.view(1, -1)
            bottom = (h - 1 - y).view(-1, 1)
            right = (w - 1 - x).view(1, -1)
            
            dist = torch.min(
                torch.min(top, bottom),
                torch.min(left, right)
            )
            
            feather = (dist < feathering).float()
            feather *= dist / feathering
            feather = torch.where(dist < feathering, feather * feather, torch.ones_like(feather))
            
            mask *= feather
            
        return mask

    def rotate_and_zoom(self, image, angle_start, angle_end, scale_start, scale_end,
                       center_x, center_y, feathering, sampling_mode, anim_start, anim_end,
                       coordinates=None):
        batch_size, height, width, channels = image.shape
        device = image.device

        # Parse coordinates if provided
        centers = None
        if coordinates is not None:
            centers = self.parse_coordinates(coordinates)

        # Prepare masks
        base_mask = self.create_base_mask(height, width, feathering, device)
        base_attn = self.create_base_mask(height, width, 0, device)

        interpolation = {
            "nearest": TF.InterpolationMode.NEAREST,
            "bilinear": TF.InterpolationMode.BILINEAR,
            "bicubic": TF.InterpolationMode.BICUBIC
        }[sampling_mode]

        rotated_images = []
        masks = []
        attn_masks = []

        for i in range(batch_size):
            # Calculate progress
            t = i / (batch_size - 1) if batch_size > 1 else 0
            progress = self.bezier(t, anim_start, anim_end)
            
            # Interpolate parameters
            current_angle = angle_start + (angle_end - angle_start) * progress
            current_scale = scale_start + (scale_end - scale_start) * progress
            
            # Get frame-specific center
            center = self.get_center_for_frame(i, centers, center_x, center_y, width, height)
            
            # Process image
            img = image[i].permute(2, 0, 1)  # HWC to CHW format
            rotated = TF.affine(
                img, 
                angle=current_angle,
                translate=[0, 0],
                scale=current_scale,
                shear=0,
                interpolation=interpolation,
                center=center,
                fill=0
            )
            
            # Process masks
            mask_rotated = TF.affine(
                base_mask,
                angle=current_angle,
                translate=[0, 0],
                scale=current_scale,
                shear=0,
                interpolation=TF.InterpolationMode.BILINEAR,
                center=center,
                fill=0
            )
            
            attn_rotated = TF.affine(
                base_attn,
                angle=current_angle,
                translate=[0, 0],
                scale=current_scale,
                shear=0,
                interpolation=TF.InterpolationMode.NEAREST,
                center=center,
                fill=0
            )
            
            # Convert back to HWC format and append
            rotated_images.append(rotated.permute(1, 2, 0))
            masks.append(mask_rotated.squeeze(0))
            attn_masks.append(attn_rotated.squeeze(0))

        # Stack results and ensure dimensions match
        rotated_images = torch.stack(rotated_images)
        masks = torch.stack(masks)
        attn_masks = torch.stack(attn_masks)
        
        return (rotated_images, 1.0 - masks, attn_masks)

NODE_CLASS_MAPPINGS = {
    "AnimatedRotationZoom": AnimatedRotationZoom
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimatedRotationZoom": "Animated Rotation & Zoom"
}