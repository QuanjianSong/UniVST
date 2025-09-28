import torch
import cv2
import numpy as np
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import matplotlib.pyplot as plt


def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image, device):
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image.unsqueeze(0).to(device)

def compute_optical_flow(model, image1, image2):
    with torch.no_grad():
        flow = model(image1, image2)
    return flow

def compute_occlusion_mask(forward_flow_np, backward_flow_np, threshold=1.0):
    h, w, _ = forward_flow_np.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    coords2 = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    coords1_reconstructed = coords2 + forward_flow_np
    coords1_back = coords1_reconstructed + backward_flow_np
    error = np.linalg.norm(coords1_back - coords2, axis=-1)
    
    occlusion_mask = (error > threshold).astype(np.uint8) * 255
    return occlusion_mask

def warp_image_with_flow(image, flow):
    h, w, _ = flow.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    coords2 = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    coords1_reconstructed = coords2 + flow

    map_x = coords1_reconstructed[..., 0].astype(np.float32)
    map_y = coords1_reconstructed[..., 1].astype(np.float32)

    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped_image

def apply_mask(image, mask, original_image):
    mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0
    masked_image = image * (1 - mask_expanded) + original_image * mask_expanded
    return masked_image.astype(np.uint8)

def post_process(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def get_warp(image1_path, image2_path, ref_image1=None, ref_image2=None, occlusion_mask_save_path=None, warped_image_save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device)
    model.eval()
    if isinstance(image1_path, str):
        image1 = load_image(image1_path)
    else:
        image1 = image1_path
    if isinstance(image2_path, str):
        image2 = load_image(image2_path)
    else:
        image2 = image2_path
    # breakpoint()
    if ref_image2 is None:
        ref_image2 = image2.copy()
    elif isinstance(ref_image2, str):
        ref_image2 = load_image(ref_image2)

    if ref_image1 is None:
        ref_image1 = image1.copy()
    elif isinstance(ref_image1, str):
        ref_image1 = load_image(ref_image1)

    image1_tensor = preprocess_image(image1, device)
    image2_tensor = preprocess_image(image2, device)

    forward_flow = compute_optical_flow(model, image1_tensor, image2_tensor)[-1]
    backward_flow = compute_optical_flow(model, image2_tensor, image1_tensor)[-1]

    forward_flow_np = forward_flow.squeeze().permute(1, 2, 0).cpu().numpy()
    backward_flow_np = backward_flow.squeeze().permute(1, 2, 0).cpu().numpy()
    # breakpoint()
    occlusion_mask = compute_occlusion_mask(forward_flow_np, backward_flow_np, threshold=1.5)

    if occlusion_mask_save_path is not None:
        cv2.imwrite(occlusion_mask_save_path, occlusion_mask)
        print(f"Occlusion mask save at {occlusion_mask_save_path}")

    warped_image = warp_image_with_flow(ref_image2, forward_flow_np)
    masked_image = apply_mask(warped_image, occlusion_mask, ref_image1)
    # post_processed_image = post_process(masked_image)
    post_processed_image = masked_image

    if warped_image_save_path is not None:
        post_processed_image_bgr = cv2.cvtColor(post_processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(warped_image_save_path, post_processed_image_bgr)
        print(f"Occlusion mask save at {warped_image_save_path}")
    return post_processed_image


if __name__ == "__main__":
    image1_path = 'The path of image1'
    image2_path = 'The path of image2'
    occlusion_mask_save_path = 'occlusion_mask.png'
    warped_image_save_path = 'warped_image_with_mask.png'
    get_warp(image1_path, image2_path, None, None, occlusion_mask_save_path, warped_image_save_path)
