import torch

class AnimatedOffsetPadding:
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

    def bezier(self, t, start_style, end_style):
        p1 = self.BEZIER_POINTS[start_style]["p1"]
        p2 = self.BEZIER_POINTS[end_style]["p2"]
        
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        
        return mt3 * 0 + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * 1

    def create_feathered_mask(self, height, width, v_offset, h_offset, new_height, new_width, feathering):
        black_region = torch.zeros((height, width), dtype=torch.float32)
        
        if feathering > 0:
            for y in range(height):
                for x in range(width):
                    dt = y if v_offset > 0 else height
                    db = height - y if (new_height - (v_offset + height)) > 0 else height
                    dl = x if h_offset > 0 else width
                    dr = width - x if (new_width - (h_offset + width)) > 0 else width

                    d = min(dt, db, dl, dr)
                    if d < feathering:
                        v = (feathering - d) / feathering
                        black_region[y, x] = v * v
                
        return black_region

    def expand_image(self, image, horiz_add, horiz_offset_start, horiz_offset_end, 
                    vert_add, vert_offset_start, vert_offset_end, feathering,
                    anim_start, anim_end):
        
        batch_size, h, w, c = image.shape
        new_height = h + vert_add
        new_width = w + horiz_add
        
        horiz_start = int(horiz_offset_start * horiz_add)
        horiz_end = int(horiz_offset_end * horiz_add)
        vert_start = int(vert_offset_start * vert_add)
        vert_end = int(vert_offset_end * vert_add)
        
        new_images = torch.ones((batch_size, new_height, new_width, c), dtype=torch.float32) * 0.5
        masks = []
        attn_masks = []
        
        for i in range(batch_size):
            progress = self.bezier(i / (batch_size - 1) if batch_size > 1 else 0, anim_start, anim_end)
            
            h_offset = int(round(horiz_start + (horiz_end - horiz_start) * progress))
            v_offset = int(round(vert_start + (vert_end - vert_start) * progress))
            
            # Create feathered mask (white outside, black inside with feathering)
            mask = torch.ones((new_height, new_width), dtype=torch.float32)
            black_region = self.create_feathered_mask(h, w, v_offset, h_offset, new_height, new_width, feathering)
            mask[v_offset:v_offset+h, h_offset:h_offset+w] = black_region
            masks.append(mask)
            
            # Create attention mask (black outside, white inside, no feathering)
            attn_mask = torch.zeros((new_height, new_width), dtype=torch.float32)
            attn_mask[v_offset:v_offset+h, h_offset:h_offset+w] = 1.0
            attn_masks.append(attn_mask)
            
            # Place image
            new_images[i, v_offset:v_offset+h, h_offset:h_offset+w, :] = image[i]
        
        return (new_images, 
                torch.stack(masks, dim=0),
                torch.stack(attn_masks, dim=0))

NODE_CLASS_MAPPINGS = {"AnimatedOffsetPadding": AnimatedOffsetPadding}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimatedOffsetPadding": "Animated Offset Padding"}