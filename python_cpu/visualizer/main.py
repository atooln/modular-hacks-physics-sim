import os
import pandas as pd
import numpy as np
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, JS, CSV)
from fastapi.responses import FileResponse

@app.get("/", response_class=FileResponse)
def get_home():
    return "static/index.html"

@app.get("/")
def get_home():
    index_path = Path("static/index.html")
    if index_path.exists():
        return HTMLResponse(index_path.read_text(), media_type="text/html")
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

@app.websocket("/ws/heatmap")
async def heatmap_ws(websocket: WebSocket):
    await websocket.accept()
    last_sent = None

    while True:
        try:
            csv_path = "static/sample.csv"
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, header=None)
                    matrix = df.values.tolist()

                    if matrix != last_sent:
                        await websocket.send_text(json.dumps(matrix))
                        last_sent = matrix
                except Exception as csv_err:
                    print("CSV read error:", csv_err)

            await asyncio.sleep(0.1)
        except Exception as e:
            print("WebSocket error:", e)
            break
