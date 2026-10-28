"""
NeRF Turntable Pipeline
=======================
Train a NeRF model on turntable images of an object and generate novel views.

Usage:
    # Using real images from a folder:
    python NeRF.py --data_dir "path/to/your/images"
    
    # Using synthetic data for testing:
    python NeRF.py --synthetic
"""

import os
import glob
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================
CONFIG = {
    'image_size': 300,        # Image resolution (balanced for quality/speed)
    'num_views': 20,          # Number of synthetic views
    'num_iters': 30000,       # Training iterations (practical for RTX 3050)
    'batch_size': 2048,       # Rays per batch (larger for GPU efficiency)
    'learning_rate': 5e-4,    # Learning rate
    'num_samples': 96,        # Samples per ray (balanced for quality/speed)
    'near': 2.0,              # Near clipping plane
    'far': 6.0,               # Far clipping plane
    'radius': 4.0,            # Camera distance from object
    'elevation_deg': 30.0,    # Camera elevation angle
}

# ==========================================================
# DEVICE SETUP
# ==========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================================
# TURNTABLE POSE GENERATOR
# ==========================================================
class TurntablePoseGenerator:
    """Generate camera poses for turntable-style captures."""
    
    def __init__(self, radius=4.0, elevation_deg=30.0):
        self.radius = radius
        self.elevation = np.radians(elevation_deg)
    
    def generate_poses(self, num_images):
        """Generate camera poses for N images in a circle."""
        poses = []
        angle_step = 2 * np.pi / num_images
        
        for i in range(num_images):
            theta = i * angle_step
            cam_x = self.radius * np.cos(self.elevation) * np.cos(theta)
            cam_y = self.radius * np.cos(self.elevation) * np.sin(theta)
            cam_z = self.radius * np.sin(self.elevation)
            camera_pos = np.array([cam_x, cam_y, cam_z])
            pose = self._look_at(camera_pos, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
            poses.append(torch.tensor(pose, dtype=torch.float32))
        return poses
    
    def _look_at(self, eye, target, up):
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        new_up = np.cross(right, forward)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = new_up
        pose[:3, 2] = -forward
        pose[:3, 3] = eye
        return pose


# ==========================================================
# IMAGE LOADER (FOR REAL IMAGES)
# ==========================================================
class ImageLoader:
    """Load real images from a folder for NeRF training."""
    
    def __init__(self, image_size=200, radius=4.0, elevation_deg=30.0):
        self.image_size = image_size
        self.radius = radius
        self.elevation_deg = elevation_deg
        self.pose_generator = TurntablePoseGenerator(radius, elevation_deg)
        self.images = []
        self.image_tensors = []
        self.poses = []
    
    def load_from_folder(self, folder_path):
        """
        Load all images from a folder.
        
        IMPORTANT: Images should be named in order (e.g., 001.jpg, 002.jpg, ...)
        and taken while walking around the object in a circle.
        """
        # Find all images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
        image_paths = sorted(image_paths)
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {folder_path}")
        
        print(f"Found {len(image_paths)} images in {folder_path}")
        
        # Load and preprocess images
        self.images = []
        self.image_tensors = []
        
        for path in tqdm(image_paths, desc="Loading images"):
            img = Image.open(path).convert('RGB')
            img = self._center_crop_and_resize(img)
            self.images.append(img)
            self.image_tensors.append(torch.tensor(np.array(img) / 255.0, dtype=torch.float32))
        
        # Generate turntable poses
        self.poses = self.pose_generator.generate_poses(len(self.images))
        
        print(f"Loaded {len(self.images)} images at {self.image_size}x{self.image_size}")
        return self.poses
    
    def _center_crop_and_resize(self, img):
        """Center crop to square and resize."""
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return img
    
    def show_samples(self, num_samples=8):
        """Display sample images."""
        n = min(num_samples, len(self.images))
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = np.atleast_2d(axes)
        
        for i in range(rows * cols):
            r, c = i // cols, i % cols
            ax = axes[r, c] if rows > 1 else axes[0, c]
            if i < n:
                ax.imshow(self.images[i])
                ax.set_title(f'Image {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('training_samples.png', dpi=100)
        print("Saved training samples to 'training_samples.png'")
        plt.show()


# ==========================================================
# SYNTHETIC DATA GENERATOR
# ==========================================================
class SyntheticDataGenerator:
    """Generate synthetic training data (colored sphere)."""
    
    def __init__(self, image_size=100, num_views=20, radius=4.0, elevation_deg=30.0):
        self.image_size = image_size
        self.num_views = num_views
        self.pose_generator = TurntablePoseGenerator(radius, elevation_deg)
        self.images = []
        self.image_tensors = []
        self.poses = []
    
    def generate(self):
        print(f"Creating synthetic dataset ({self.num_views} views at {self.image_size}x{self.image_size})...")
        self.poses = self.pose_generator.generate_poses(self.num_views)
        self.images = []
        self.image_tensors = []
        
        for i in range(self.num_views):
            theta = 2 * np.pi * i / self.num_views
            img = self._render_sphere(theta)
            self.images.append(Image.fromarray((img * 255).astype(np.uint8)))
            self.image_tensors.append(torch.tensor(img, dtype=torch.float32))
        
        print(f"Created {self.num_views} synthetic views")
        return self.poses
    
    def _render_sphere(self, theta):
        size = self.image_size
        img = np.ones((size, size, 3))
        center = size // 2
        radius = size * 0.35
        
        y_coords, x_coords = np.ogrid[:size, :size]
        dx = x_coords - center
        dy = y_coords - center
        dist = np.sqrt(dx**2 + dy**2)
        mask = dist < radius
        
        nz = np.zeros_like(dist)
        nz[mask] = np.sqrt(np.maximum(0, radius**2 - dist[mask]**2)) / radius
        nx = dx / radius
        ny = dy / radius
        
        r = 0.5 + 0.4 * np.sin(theta + nx * 2)
        g = 0.5 + 0.4 * np.cos(theta + ny * 2)
        b = 0.5 + 0.4 * np.sin(theta * 0.5 + nz * 2)
        
        light_dir = np.array([np.cos(theta), np.sin(theta), 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        shade = np.maximum(0.3, nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2])
        
        img[..., 0] = np.where(mask, r * shade, 1.0)
        img[..., 1] = np.where(mask, g * shade, 1.0)
        img[..., 2] = np.where(mask, b * shade, 1.0)
        
        return np.clip(img, 0, 1)
    
    def show_samples(self, num_samples=8):
        n = min(num_samples, len(self.images))
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = np.atleast_2d(axes)
        
        for i in range(rows * cols):
            r, c = i // cols, i % cols
            ax = axes[r, c] if rows > 1 else axes[0, c]
            if i < n:
                ax.imshow(self.images[i])
                ax.set_title(f'View {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('training_samples.png', dpi=100)
        print("Saved training samples to 'training_samples.png'")
        plt.show()


# ==========================================================
# POSITIONAL ENCODING
# ==========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('frequencies', freqs)
    
    def forward(self, x):
        x_scaled = x[..., None] * self.frequencies * math.pi
        return torch.cat([x, torch.sin(x_scaled).flatten(-2), torch.cos(x_scaled).flatten(-2)], dim=-1)
    
    def output_dim(self, input_dim):
        return input_dim * (1 + 2 * self.num_frequencies)


# ==========================================================
# NeRF NETWORK
# ==========================================================
class NeRF(nn.Module):
    def __init__(self, pos_freqs=10, dir_freqs=4, hidden_dim=512, num_layers=8):
        super().__init__()
        self.pos_encoder = PositionalEncoding(pos_freqs)
        self.dir_encoder = PositionalEncoding(dir_freqs)
        
        pos_dim = self.pos_encoder.output_dim(3)
        dir_dim = self.dir_encoder.output_dim(3)
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(pos_dim, hidden_dim))
        for i in range(num_layers - 1):
            in_dim = hidden_dim + pos_dim if i == 4 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))
        
        self.skip_layer = 5
        self.density_layer = nn.Linear(hidden_dim, 1)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.color_layer1 = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.color_layer2 = nn.Linear(hidden_dim // 2, 3)
    
    def forward(self, positions, directions):
        pos_enc = self.pos_encoder(positions)
        dir_enc = self.dir_encoder(directions)
        
        h = pos_enc
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, pos_enc], dim=-1)
            h = F.relu(layer(h))
        
        density = F.softplus(self.density_layer(h))
        features = self.feature_layer(h)
        h_color = F.relu(self.color_layer1(torch.cat([features, dir_enc], dim=-1)))
        rgb = torch.sigmoid(self.color_layer2(h_color))
        
        return rgb, density


# ==========================================================
# VOLUMETRIC RENDERER
# ==========================================================
class VolumetricRenderer:
    def __init__(self, near=2.0, far=6.0, num_samples=64):
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.training = True
    
    def generate_rays(self, H, W, focal, pose):
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=pose.device),
            torch.arange(H, dtype=torch.float32, device=pose.device),
            indexing='xy'
        )
        dirs = torch.stack([(i - W/2) / focal, -(j - H/2) / focal, -torch.ones_like(i)], dim=-1)
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        rays_d = F.normalize(rays_d, dim=-1)
        rays_o = pose[:3, 3].expand(rays_d.shape)
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    
    def sample_points(self, rays_o, rays_d):
        N = rays_o.shape[0]
        t_vals = torch.linspace(0, 1, self.num_samples, device=rays_o.device)
        z_vals = self.near + (self.far - self.near) * t_vals
        z_vals = z_vals.expand(N, self.num_samples)
        
        if self.training:
            z_vals = z_vals + torch.rand_like(z_vals) * (self.far - self.near) / self.num_samples
        
        points = rays_o[:, None, :] + z_vals[:, :, None] * rays_d[:, None, :]
        return points, z_vals
    
    def render(self, rgb, density, z_vals):
        deltas = torch.cat([z_vals[:, 1:] - z_vals[:, :-1], 1e10 * torch.ones_like(z_vals[:, :1])], dim=-1)
        if density.dim() == 3:
            density = density.squeeze(-1)
        
        alpha = 1.0 - torch.exp(-density * deltas)
        T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        weights = T * alpha
        
        color = torch.sum(weights[:, :, None] * rgb, dim=1)
        depth = torch.sum(weights * z_vals, dim=1)
        acc = torch.sum(weights, dim=1, keepdim=True)
        color = color + (1 - acc)
        
        return color, depth


# ==========================================================
# TRAINER
# ==========================================================
class NeRFTrainer:
    def __init__(self, model, renderer, images, poses, lr=5e-4, batch_size=1024):
        self.model = model
        self.renderer = renderer
        self.batch_size = batch_size
        self.H, self.W = images[0].shape[:2]
        self.focal = self.W
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        print("Preparing rays...")
        all_rays_o, all_rays_d, all_colors = [], [], []
        
        for img, pose in zip(images, poses):
            pose = pose.to(device)
            rays_o, rays_d = renderer.generate_rays(self.H, self.W, self.focal, pose)
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)
            all_colors.append(img.reshape(-1, 3).to(device))
        
        self.rays_o = torch.cat(all_rays_o, 0)
        self.rays_d = torch.cat(all_rays_d, 0)
        self.colors = torch.cat(all_colors, 0)
        print(f"Prepared {len(self.rays_o):,} rays")
    
    def train_step(self):
        self.model.train()
        self.renderer.training = True
        
        idx = torch.randint(0, len(self.rays_o), (self.batch_size,), device=device)
        rays_o, rays_d, target = self.rays_o[idx], self.rays_d[idx], self.colors[idx]
        
        points, z_vals = self.renderer.sample_points(rays_o, rays_d)
        N, S = points.shape[:2]
        
        rgb, density = self.model(points.reshape(-1, 3), rays_d[:, None, :].expand(N, S, 3).reshape(-1, 3))
        color_pred, _ = self.renderer.render(rgb.reshape(N, S, 3), density.reshape(N, S, 1), z_vals)
        
        loss = F.mse_loss(color_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_iters=2000):
        losses = []
        pbar = tqdm(range(num_iters), desc="Training")
        
        for i in pbar:
            loss = self.train_step()
            losses.append(loss)
            if i % 100 == 0:
                pbar.set_postfix({'loss': f'{loss:.6f}'})
        
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=100)
        print("Saved training loss to 'training_loss.png'")
        plt.show()
        
        return losses
    
    def render_view(self, pose, chunk_size=4096):
        self.model.eval()
        self.renderer.training = False
        pose = pose.to(device)
        
        rays_o, rays_d = self.renderer.generate_rays(self.H, self.W, self.focal, pose)
        all_colors = []
        
        with torch.no_grad():
            for i in range(0, len(rays_o), chunk_size):
                ro, rd = rays_o[i:i+chunk_size], rays_d[i:i+chunk_size]
                points, z_vals = self.renderer.sample_points(ro, rd)
                N, S = points.shape[:2]
                rgb, density = self.model(points.reshape(-1, 3), rd[:, None, :].expand(N, S, 3).reshape(-1, 3))
                color, _ = self.renderer.render(rgb.reshape(N, S, 3), density.reshape(N, S, 1), z_vals)
                all_colors.append(color)
        
        return torch.cat(all_colors, 0).reshape(self.H, self.W, 3).cpu().numpy()


# ==========================================================
# VISUALIZATION
# ==========================================================
def compare_views(trainer, data, num_views=4):
    """Compare ground truth vs rendered views."""
    num_views = min(num_views, len(data.images))
    fig, axes = plt.subplots(2, num_views, figsize=(4*num_views, 8))
    
    for i in range(num_views):
        axes[0, i].imshow(data.image_tensors[i].numpy())
        axes[0, i].set_title(f'Ground Truth {i+1}')
        axes[0, i].axis('off')
        
        rendered = trainer.render_view(data.poses[i])
        axes[1, i].imshow(np.clip(rendered, 0, 1))
        axes[1, i].set_title(f'Rendered {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=100)
    print("Saved comparison to 'comparison.png'")
    plt.show()


def render_novel_views(trainer, pose_generator, num_views=16):
    """Render novel views from angles NOT in the training set."""
    # Generate more poses than training to get novel intermediate views
    poses = pose_generator.generate_poses(num_views)
    
    cols = 4
    rows = (num_views + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_2d(axes)
    
    print(f"Rendering {num_views} novel views...")
    for i, pose in enumerate(tqdm(poses, desc="Rendering novel views")):
        r, c = i // cols, i % cols
        ax = axes[r, c] if rows > 1 else axes[0, c]
        rendered = trainer.render_view(pose)
        ax.imshow(np.clip(rendered, 0, 1))
        angle = int(360 * i / num_views)
        ax.set_title(f'{angle}°')
        ax.axis('off')
    
    for i in range(num_views, rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c] if rows > 1 else axes[0, c]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('novel_views.png', dpi=150)
    print("Saved novel views to 'novel_views.png'")
    plt.show()


def render_360_gif(trainer, pose_generator, num_frames=60, output_path='360_view.gif'):
    """Create a smooth 360° rotation animation."""
    try:
        import imageio
    except ImportError:
        print("Install imageio with: pip install imageio")
        return
    
    poses = pose_generator.generate_poses(num_frames)
    frames = []
    
    print(f"Rendering {num_frames} frames for 360° animation...")
    for pose in tqdm(poses, desc="Rendering animation"):
        frame = trainer.render_view(pose)
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frames.append(frame)
    
    imageio.mimsave(output_path, frames, fps=15, loop=0)
    print(f"Saved 360° animation to '{output_path}'")


# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description='NeRF Turntable Pipeline')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to folder containing images of your object')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--image_size', type=int, default=CONFIG['image_size'],
                        help='Image resolution')
    parser.add_argument('--num_iters', type=int, default=CONFIG['num_iters'],
                        help='Training iterations')
    parser.add_argument('--radius', type=float, default=CONFIG['radius'],
                        help='Camera distance from object')
    parser.add_argument('--elevation', type=float, default=CONFIG['elevation_deg'],
                        help='Camera elevation angle in degrees')
    parser.add_argument('--render_gif', action='store_true',
                        help='Render 360° GIF animation')
    args = parser.parse_args()
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("=" * 60)
    print("NeRF Turntable Pipeline")
    print("=" * 60)
    
    # Load data
    if args.data_dir:
        print(f"\nLoading images from: {args.data_dir}")
        data = ImageLoader(
            image_size=args.image_size,
            radius=args.radius,
            elevation_deg=args.elevation
        )
        data.load_from_folder(args.data_dir)
    else:
        print("\nUsing synthetic data (use --data_dir to train on real images)")
        data = SyntheticDataGenerator(
            image_size=args.image_size,
            num_views=CONFIG['num_views'],
            radius=args.radius,
            elevation_deg=args.elevation
        )
        data.generate()
    
    data.show_samples()
    
    # Create model
    model = NeRF().to(device)
    print(f"\nNeRF model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create renderer and trainer
    renderer = VolumetricRenderer(
        near=CONFIG['near'],
        far=CONFIG['far'],
        num_samples=CONFIG['num_samples']
    )
    
    trainer = NeRFTrainer(
        model=model,
        renderer=renderer,
        images=data.image_tensors,
        poses=data.poses,
        lr=CONFIG['learning_rate'],
        batch_size=CONFIG['batch_size']
    )
    
    # Train
    print(f"\nTraining for {args.num_iters} iterations...")
    trainer.train(num_iters=args.num_iters)
    
    # Compare training views
    print("\nComparing ground truth vs rendered...")
    compare_views(trainer, data, num_views=4)
    
    # Render novel views (views NOT in training set)
    print("\nRendering novel views (unseen angles)...")
    render_novel_views(trainer, data.pose_generator, num_views=16)
    
    # Optional: Render 360° GIF
    if args.render_gif:
        render_360_gif(trainer, data.pose_generator, num_frames=60)
    
    # Save model
    torch.save(model.state_dict(), 'nerf_model.pth')
    print("\nModel saved to 'nerf_model.pth'")
    
    print("\n" + "=" * 60)
    print("Done! Output files:")
    print("  - training_samples.png: Input images")
    print("  - training_loss.png: Loss curve")
    print("  - comparison.png: Ground truth vs rendered")
    print("  - novel_views.png: Novel synthesized views (unseen angles)")
    if args.render_gif:
        print("  - 360_view.gif: 360° rotation animation")
    print("=" * 60)


if __name__ == "__main__":
    main()
