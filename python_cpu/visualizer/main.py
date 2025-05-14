import os
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio
import glob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def get_home():
    index_path = Path("static/index.html")
    if index_path.exists():
        return HTMLResponse(index_path.read_text(), media_type="text/html")
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.get("/heatmap.png")
def get_latest_png():
    return FileResponse("static/heatmap.png", media_type="image/png")

# Background task to monitor new step_*.csv files and render PNG
@app.on_event("startup")
async def watch_for_csv():
    print("Watcher starting...")
    seen_files = set()

    while True:
        try:
            csv_files = sorted(glob.glob("static/output1/step_*.csv"))
            new_files = [f for f in csv_files if f not in seen_files]

            if new_files:
                latest = new_files[-1]
                print(f"Rendering: {latest}")
                df = pd.read_csv(latest, header=None)
                save_heatmap(df.values)

                seen_files.update(new_files)

        except Exception as e:
            print("Watcher error:", e)

        await asyncio.sleep(0.5)

def save_heatmap(matrix, out_path="static/heatmap.png"):
    plt.imshow(matrix, cmap="viridis", origin="lower", vmin=0, vmax=1)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
