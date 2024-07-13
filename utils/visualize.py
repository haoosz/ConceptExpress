import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def get_boundary_and_eroded_mask(masks, colors):
    kernel = np.ones((7, 7), np.uint8)
    eroded_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    boundary_mask = np.zeros((256, 256, 3), dtype=np.uint8)

    arr = np.zeros((256, 256, 3), dtype=np.uint8) # all-zero array
    
    for i, part_mask in enumerate(masks):
        part_mask_erosion = cv2.erode(
            part_mask.astype(np.uint8), kernel, iterations=1
            )[..., None].astype(np.uint8)
 
        part_boundary_mask = part_mask.astype(np.uint8) - part_mask_erosion
        
        pic = (colors[i] - arr).astype(np.uint8)
        
        erosion = pic * part_mask_erosion
        boundary = pic * part_boundary_mask
        
        eroded_mask += erosion
        boundary_mask += boundary

    return eroded_mask, boundary_mask
    
def vis_masked_image(image, masks, colors, instance_level=True):
    tsfm = transforms.Resize([256,256])
    image_np = np.array(tsfm(image))
    
    
    tsfm_mask = transforms.Resize([256,256], interpolation=transforms.InterpolationMode.NEAREST)
    
    masks_np = []
    mask_all = np.zeros((256, 256, 3), dtype=np.uint8) # all-zero array
    
    for m in masks:
        m_np = np.array(tsfm_mask(m)).reshape(256,256,1) / 255
        masks_np.append(m_np)
        mask_all += m_np.astype(np.uint8)
        
    eroded_mask, boundary_mask = get_boundary_and_eroded_mask(masks_np, colors)
    
    if instance_level:
        image_masked = ((boundary_mask * 0.6 + eroded_mask * 0.4 + image_np * 0.4) * mask_all +
                        (image_np * 0.2) * (1-mask_all))
    else:
        image_masked = ((boundary_mask * 0.6 + eroded_mask * 0.6 + image_np * 0.4) * mask_all +
                        (image_np * 0.2) * (1-mask_all))
    
    # image_masked = image_np * mask_np.astype(np.uint8)
    
    image_masked = Image.fromarray(image_masked.astype(np.uint8))
    
    return image_masked