"""
Generate 360° GIF from a trained NeRF model.
"""
import torch
import numpy as np
from tqdm import tqdm
import imageio
# Import from main NeRF file
from NeRF import (
    NeRF, VolumetricRenderer, TurntablePoseGenerator, 
    CONFIG, device
)

def generate_gif():
    # Create model with same architecture as trained (hidden_dim=256)
    model = NeRF(hidden_dim=256).to(device)
    model.load_state_dict(torch.load('nerf_model.pth', map_location=device))
    model.eval()

    
    # Create renderer
    renderer = VolumetricRenderer(
        near=CONFIG['near'],
        far=CONFIG['far'],
        num_samples=CONFIG['num_samples']
    )
    renderer.training = False
    
    # Setup
    H = W = CONFIG['image_size']
    focal = W
    pose_generator = TurntablePoseGenerator(
        radius=CONFIG['radius'],
        elevation_deg=CONFIG['elevation_deg']
    )
    
    # Generate 60 frames for smooth animation
    num_frames = 60
    poses = pose_generator.generate_poses(num_frames)
    frames = []
    
    
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
    
    # Save GIF
    output_path = '360_view.gif'
    imageio.mimsave(output_path, frames, fps=15, loop=0)
    

