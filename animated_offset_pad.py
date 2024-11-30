import torch
import cv2
import numpy as np

class AnimatedOffsetPadding:
    MAX_RESOLUTION = 16384
    
    # Pre-computed bezier control points for different animation curves
    BEZIER_POINTS = {
        "linear": {"p1": np.array([0.4, 0.4]), "p2": np.array([0.6, 0.6])},
        "fast":   {"p1": np.array([0.1, 0.8]), "p2": np.array([0.9, 0.2])},
        "slow":   {"p1": np.array([0.8, 0.1]), "p2": np.array([0.2, 0.9])},
    }

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "horiz_add": ("INT", {"default": 512, "min": 0, "max": s.MAX_RESOLUTION, "step": 8}),
            "horiz_offset_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "horiz_offset_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "vert_add": ("INT", {"default": 64, "min": 0, "max": s.MAX_RESOLUTION, "step": 8}),
            "vert_offset_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "vert_offset_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "feathering": ("INT", {"default": 40, "min": 0, "max": s.MAX_RESOLUTION, "step": 1}),
            "anim_start": (["linear", "fast", "slow"],),
            "anim_end": (["linear", "fast", "slow"],),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK", "ATTN_MASK")
    FUNCTION = "expand_image"
    CATEGORY = "image"

    @staticmethod
    def bezier(t, start_style, end_style):
        p1 = AnimatedOffsetPadding.BEZIER_POINTS[start_style]["p1"][1]
        p2 = AnimatedOffsetPadding.BEZIER_POINTS[end_style]["p2"][1]
        
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        
        return 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3

    def create_masks(self, height, width, v_offset, h_offset, new_height, new_width, feathering, device):
        # Create base mask on CPU
        base_mask = torch.ones((new_height, new_width), dtype=torch.float32, device='cpu')
        base_mask[v_offset:v_offset+height, h_offset:h_offset+width] = 0.0
        attn_mask = 1.0 - base_mask

        if feathering > 0:
            dx = dy = feathering * 2 + 1
            mask = base_mask.numpy()
            mask = cv2.GaussianBlur(mask, (dx, dy), 0, cv2.BORDER_REPLICATE) * 2
            mask = torch.from_numpy(mask)
            mask[base_mask == 1.0] = 1.0
        else:
            mask = base_mask

        return mask.to(device), attn_mask.to(device)

    def expand_image(self, image, horiz_add, horiz_offset_start, horiz_offset_end,
                    vert_add, vert_offset_start, vert_offset_end, feathering,
                    anim_start, anim_end):
        
        batch_size, h, w, c = image.shape
        device = image.device

        if h + vert_add > self.MAX_RESOLUTION or w + horiz_add > self.MAX_RESOLUTION:
            raise ValueError(f"Output dimensions would exceed maximum resolution of {self.MAX_RESOLUTION}")
        
        new_height = h + vert_add
        new_width = w + horiz_add
        
        # Calculate offset ranges
        horiz_start = int(horiz_offset_start * horiz_add)
        horiz_end = int(horiz_offset_end * horiz_add)
        vert_start = int(vert_offset_start * vert_add)
        vert_end = int(vert_offset_end * vert_add)
        
        # Initialize output tensors
        new_images = torch.ones((batch_size, new_height, new_width, c), 
                              dtype=torch.float32, device=device) * 0.5
        masks = []
        attn_masks = []
        
        for i in range(batch_size):
            progress = self.bezier(i / (batch_size - 1) if batch_size > 1 else 0, 
                                 anim_start, anim_end)
            
            h_offset = int(round(horiz_start + (horiz_end - horiz_start) * progress))
            v_offset = int(round(vert_start + (vert_end - vert_start) * progress))
            
            mask, attn_mask = self.create_masks(
                h, w, v_offset, h_offset, new_height, new_width, feathering, device
            )
            
            masks.append(mask)
            attn_masks.append(attn_mask)
            new_images[i, v_offset:v_offset+h, h_offset:h_offset+w, :] = image[i]
        
        return (new_images, 
                torch.stack(masks, dim=0),
                torch.stack(attn_masks, dim=0))

NODE_CLASS_MAPPINGS = {"AnimatedOffsetPadding": AnimatedOffsetPadding}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimatedOffsetPadding": "Animated Offset Padding"}