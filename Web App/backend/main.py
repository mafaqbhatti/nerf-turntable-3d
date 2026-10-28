import os
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import get_db, SessionLocal, Base, engine
from models import RenderJob, RenderStatus
from schemas import (
    RenderJobResponse,
    RenderJobStatusResponse,
    RenderJobListResponse,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="NeRF Render API",
    description="API for generating 360° GIF animations from NeRF models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_gif_generation(job_id: str):
    db = SessionLocal()
    try:
        job = db.query(RenderJob).filter(RenderJob.id == job_id).first()
        if not job:
            print(f"[ERROR] Job {job_id} not found")
            return

        print(f"[INFO] Starting render job {job_id}")
        job.status = RenderStatus.PROCESSING
        job.progress = 10
        db.commit()

        from generate_gif import generate_gif

        print(f"[INFO] Calling generate_gif()...")
        job.progress = 20
        db.commit()

        generate_gif()

        print(f"[INFO] GIF generation completed!")
        job.status = RenderStatus.COMPLETED
        job.progress = 100
        job.gif_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "sample.gif")
        )
        db.commit()
        print(f"[INFO] Job {job_id} completed successfully. GIF at: {job.gif_path}")

    except Exception as e:
        print(f"[ERROR] Job {job_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        job = db.query(RenderJob).filter(RenderJob.id == job_id).first()
        if job:
            job.status = RenderStatus.FAILED
            job.error_msg = str(e)
            db.commit()
    finally:
        db.close()


@app.get("/")
async def root():
    """Serve the React frontend"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "API is running. Build React app and place in 'static' folder."}


@app.post("/render", response_model=RenderJobResponse)
async def start_render(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    job = RenderJob(
        status=RenderStatus.NOT_STARTED,
        progress=0
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(run_gif_generation, job.id)

    return job


@app.get("/render/{job_id}", response_model=RenderJobResponse)
async def get_render_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(RenderJob).filter(RenderJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Render job not found")
    return job


@app.get("/render/{job_id}/status", response_model=RenderJobStatusResponse)
async def get_render_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(RenderJob).filter(RenderJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Render job not found")
    return job


@app.get("/render/{job_id}/download")
async def download_gif(job_id: str, db: Session = Depends(get_db)):
    print(f"[DOWNLOAD] Request for job: {job_id}")
    job = db.query(RenderJob).filter(RenderJob.id == job_id).first()

    if not job:
        print(f"[DOWNLOAD] Job not found in database")
        raise HTTPException(status_code=404, detail="Render job not found")

    print(f"[DOWNLOAD] Job status: {job.status}, gif_path: {job.gif_path}")
    
    if job.status != RenderStatus.COMPLETED:
        print(f"[DOWNLOAD] Job not completed yet")
        raise HTTPException(status_code=400, detail="Render job not completed")

    if not job.gif_path:
        print(f"[DOWNLOAD] gif_path is None or empty")
        raise HTTPException(status_code=404, detail="GIF path not set")
    
    if not os.path.exists(job.gif_path):
        print(f"[DOWNLOAD] GIF file does not exist at: {job.gif_path}")
        raise HTTPException(status_code=404, detail=f"GIF not found at {job.gif_path}")

    print(f"[DOWNLOAD] Serving GIF from: {job.gif_path}")
    return FileResponse(
        job.gif_path,
        media_type="image/gif",
        filename="sample.gif"
    )


@app.get("/renders", response_model=RenderJobListResponse)
async def list_render_jobs(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    jobs = (
        db.query(RenderJob)
        .order_by(RenderJob.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    total = db.query(RenderJob).count()
    return RenderJobListResponse(jobs=jobs, total=total)


@app.delete("/render/{job_id}")
async def delete_render_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(RenderJob).filter(RenderJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Render job not found")

    if job.gif_path and os.path.exists(job.gif_path):
        os.remove(job.gif_path)

    db.delete(job)
    db.commit()

    return {"status": "deleted", "job_id": str(job_id)}


# --- STATIC FILE SERVING FOR REACT FRONTEND ---
# The static folder should contain the built React app (output from 'npm run build')
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Mount static assets (JS, CSS, images, etc.) - only if the directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Catch-all route to serve the React SPA for client-side routing"""
    static_file = os.path.join(STATIC_DIR, full_path)
    
    # If requesting a static file that exists, serve it
    if full_path and os.path.exists(static_file) and os.path.isfile(static_file):
        return FileResponse(static_file)
    
    # Otherwise serve index.html for SPA routing
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    # If no static folder exists, return API info
    return {"message": "NeRF API is running. Build the React app and place it in the 'static' folder."}
