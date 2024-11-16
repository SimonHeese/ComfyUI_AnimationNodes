import torch

MAX_RESOLUTION = 16384

class AnimatedOffsetPadding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "horiz_add": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "horiz_offset_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "horiz_offset_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vert_add": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "vert_offset_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vert_offset_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1}), 
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    CATEGORY = "image"

    def expand_image(self, image, horiz_add, horiz_offset_start, horiz_offset_end, 
                    vert_add, vert_offset_start, vert_offset_end, feathering):
        
        batch_size = image.shape[0]
        
        b, h, w, c = image.shape
        
        new_height = h + vert_add
        new_width = w + horiz_add
        
        
        horiz_start = int(horiz_offset_start * horiz_add)
        horiz_end = int(horiz_offset_end * horiz_add)
        vert_start = int(vert_offset_start * vert_add)
        vert_end = int(vert_offset_end * vert_add)
        
        new_images = torch.ones((batch_size, new_height, new_width, c), dtype=torch.float32) * 0.5
        masks = []
        

        for i in range(batch_size):
            
            progress = i / (batch_size - 1) if batch_size > 1 else 0
            
            h_offset = int(round(horiz_start + (horiz_end - horiz_start) * progress))
            v_offset = int(round(vert_start + (vert_end - vert_start) * progress))
            
            mask = torch.ones((new_height, new_width), dtype=torch.float32)
            inner_mask = torch.zeros((h, w), dtype=torch.float32)
            
            if feathering > 0 :
                              
                for y in range(h):
                    for x in range(w):
                        dt = y if v_offset > 0 else h
                        db = h - y if (new_height - (v_offset + h)) > 0 else h
                        dl = x if h_offset > 0 else w
                        dr = w - x if (new_width - (h_offset + w)) > 0 else w

                        d = min(dt, db, dl, dr)

                        if d >= feathering:
                            continue

                        v = (feathering - d) / feathering
                        inner_mask[y, x] = v * v

                        

            new_images[i, v_offset:v_offset+h, h_offset:h_offset+w, :] = image[i]
            mask[v_offset:v_offset+h, h_offset:h_offset+w] = inner_mask
            masks.append(mask)
            
        final_masks = torch.stack(masks, dim=0)
            
        return (new_images, final_masks)

NODE_CLASS_MAPPINGS = {
    "AnimatedOffsetPadding": AnimatedOffsetPadding
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimatedOffsetPadding": "Animated Offset Padding"
}