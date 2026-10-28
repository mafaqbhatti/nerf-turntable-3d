# NeRF Turntable — 2D Photos to 3D Neural Rendering

A from-scratch PyTorch implementation of **Neural Radiance Fields (NeRF)** that turns a set of turntable photos of an object into a trained neural model capable of rendering brand-new, unseen 360° views — wrapped in a full-stack web app (FastAPI + React) so anyone can generate a rotating GIF from the trained model with one click.

![status](https://img.shields.io/badge/status-research%20prototype-blue)
![python](https://img.shields.io/badge/python-3.11-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.x-orange)

## Demo

`Web App/backend/sample.gif` is an example 360° render produced by the trained model included in this repo.

## What is a NeRF?

A Neural Radiance Field represents a 3D scene not as a mesh or point cloud, but as a small neural network that answers: *"at 3D point (x, y, z), viewed from direction (dx, dy, dz), what color and density is here?"*

To render a new view, you shoot a ray through each pixel, sample points along it, query the network at each point, and combine the results with classic volume-rendering math (accumulating color/opacity front-to-back, like light passing through fog). Because the network learns a continuous function of space rather than memorizing pixels, it can synthesize believable views from camera angles it never saw during training.

## How this project works

### 1. `Model/` — the NeRF engine (`NeRF.py`)

| Component | What it does |
|---|---|
| `TurntablePoseGenerator` | Since there's no camera-calibration data, this assumes photos were taken walking in a circle around the object and computes a camera pose (position + look-at orientation) for each shot. |
| `ImageLoader` | Loads real photos from `Model/data/`, center-crops to square, resizes to 300×300. |
| `SyntheticDataGenerator` | Procedurally renders a fake shaded sphere from N angles — a way to sanity-check the pipeline without real photos. |
| `PositionalEncoding` | Expands raw (x, y, z) and view-direction values into sin/cos waves at increasing frequencies, so the MLP can represent fine detail (the key trick from the original NeRF paper). |
| `NeRF` (the network) | An 8-layer, 512-unit MLP. Encoded position → density + a feature vector; feature + encoded view direction → 2 more layers → RGB color. Splitting density from color like this lets the model handle view-dependent shading. |
| `VolumetricRenderer` | Samples 96 points per ray between near/far planes, queries the network, and alpha-composites the results into one pixel color (`generate_rays`, `sample_points`, `render`). |
| `NeRFTrainer` | Pre-computes all rays/target colors, then trains via random batches of rays (default 2048) for up to 30,000 iterations, minimizing MSE between rendered and true pixel color. |
| `compare_views` / `render_novel_views` / `render_360_gif` | Visualization utilities: ground-truth vs. rendered comparison, grid of unseen angles, and the smooth rotating GIF. |

Run it directly:

```bash
cd Model

# Train on your own turntable photos (name them 001.jpg, 002.jpg, ... in shooting order)
python NeRF.py --data_dir "./data" --render_gif

# Or test the pipeline with a synthetic sphere (no real photos needed)
python NeRF.py --synthetic --render_gif
```

Useful flags: `--image_size`, `--num_iters`, `--radius`, `--elevation` (camera distance/height assumptions for the turntable pose generator).

Outputs: `training_samples.png`, `training_loss.png`, `comparison.png`, `novel_views.png`, `360_view.gif`, and the trained weights `nerf_model.pth`.

### 2. `Web App/` — serving the trained model

- **`backend/`** — FastAPI app with a small render-job queue backed by SQLite:
  - `POST /render` — kicks off GIF generation as a background task, returns a job id
  - `GET /render/{id}/status` — poll job progress (`not_started` → `processing` → `completed`/`failed`)
  - `GET /render/{id}/download` — stream back the finished GIF
  - `generate_gif.py` loads `nerf_model.pth` and reuses the same rendering loop as `Model/NeRF.py` to produce `sample.gif`.
  - Serves the built React app as static files, so the whole thing runs as a single server.
- **`frontend/`** — React + Vite + Tailwind single-page UI. One button ("Launch Render Job") triggers a render, polls status every 2 seconds, then shows and lets you download the resulting GIF.

Run it locally:

```bash
# Backend
cd "Web App/backend"
pip install -r ../../requirements.txt
cp .env.example .env   # fill in your own values
uvicorn main:app --reload

# Frontend (separate terminal, for development)
cd "Web App/frontend"
npm install
npm run dev
```

For production, build the frontend and drop it into the backend's static folder so one server handles everything:

```bash
cd "Web App/frontend"
npm run build
cp -r dist/* "../backend/static/"
```

## Project structure

```
├── Model/
│   ├── NeRF.py            # training + rendering pipeline (the actual NeRF)
│   ├── generate_gif.py    # standalone script: load a trained model, render a GIF
│   ├── data/               # 20 turntable photos used for training
│   └── nerf_model.pth      # trained weights
├── Web App/
│   ├── backend/            # FastAPI: render job API + static file serving
│   │   ├── main.py
│   │   ├── models.py / schemas.py / database.py
│   │   ├── generate_gif.py
│   │   └── nerf_model.pth
│   └── frontend/           # React + Vite + Tailwind UI
│       └── src/pages/Homepage.jsx
└── requirements.txt
```

## Requirements

- Python 3.11, PyTorch 2.x (CUDA optional but recommended — this was tuned for an RTX 3050)
- Node.js + npm (for the frontend)
- See `requirements.txt` for exact Python packages

## Known quirks / things to clean up

- `Web App/backend/generate_gif.py` has a leftover hardcoded Windows path (`MODEL_DIR = r"e:\NERF-V1\NERF-V1\Model"`) from development — it's unused at runtime (the model loads from the backend folder itself) but should be removed.
- The backend defaults to SQLite (`nerf_app.db`) for zero-config local development; a Postgres connection string is present in `database.py` (commented out) if you want to switch.
- `Model/generate_gif.py` and `Web App/backend/generate_gif.py` are two versions of the same script — the backend one is standalone and doesn't depend on `Model/`.

## Credits

Built as a project for the **CAIS Society** (Center for Artificial Intelligent Systems). Implementation follows the architecture from *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis* (Mildenhall et al., 2020).
