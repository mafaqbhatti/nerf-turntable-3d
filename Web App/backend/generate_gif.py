import sys
import os

# Set up paths
BACKEND_DIR = os.path.dirname(__file__)
MODEL_DIR = r"e:\NERF-V1\NERF-V1\Model"
sys.path.insert(0, MODEL_DIR)

import torch
import numpy as np
from tqdm import tqdm
import imageio
from NeRF import (
    NeRF, VolumetricRenderer, TurntablePoseGenerator, 
    CONFIG, device
)

def generate_gif():
    print("[generate_gif] Starting...")
    
    model = NeRF(hidden_dim=256).to(device)
    
    # Load model from the backend folder (where user copied it)
    model_path = os.path.join(BACKEND_DIR, 'nerf_model.pth')
    print(f"[generate_gif] Loading model from: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[generate_gif] Model loaded successfully")

    renderer = VolumetricRenderer(
        near=CONFIG['near'],
        far=CONFIG['far'],
        num_samples=CONFIG['num_samples']
    )
    renderer.training = False

    H = W = CONFIG['image_size']
    focal = W
    pose_generator = TurntablePoseGenerator(
        radius=CONFIG['radius'],
        elevation_deg=CONFIG['elevation_deg']
    )

    num_frames = 60
    poses = pose_generator.generate_poses(num_frames)
    frames = []

    print(f"[generate_gif] Rendering {num_frames} frames...")
    for pose in tqdm(poses, desc="Rendering"):
        pose = pose.to(device)
        rays_o, rays_d = renderer.generate_rays(H, W, focal, pose)
        all_colors = []

        chunk_size = 4096
        with torch.no_grad():
            for i in range(0, len(rays_o), chunk_size):
                ro, rd = rays_o[i:i+chunk_size], rays_d[i:i+chunk_size]
                points, z_vals = renderer.sample_points(ro, rd)
                N, S = points.shape[:2]
                rgb, density = model(
                    points.reshape(-1, 3), 
                    rd[:, None, :].expand(N, S, 3).reshape(-1, 3)
                )
                color, _ = renderer.render(
                    rgb.reshape(N, S, 3), 
                    density.reshape(N, S, 1), 
                    z_vals
                )
                all_colors.append(color)

        frame = torch.cat(all_colors, 0).reshape(H, W, 3).cpu().numpy()
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frames.append(frame)

    # Save output in the backend folder
    output_path = os.path.join(BACKEND_DIR, 'sample.gif')
    print(f"[generate_gif] Saving GIF to: {output_path}")
    imageio.mimsave(output_path, frames, fps=15, loop=0)
    print("[generate_gif] GIF saved successfully!")
