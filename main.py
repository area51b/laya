from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
import laya, os
os.environ["USE_TF"] = "0"

app = FastAPI()
router = laya.Router(preload=True, device="cpu")  # for mac, use "cpu" or "cuda" on the NVIDIA GPU

BASE_DIR = Path(__file__).resolve().parent
DINO_HTML = BASE_DIR / "dino.html"


@app.get("/")
@app.get("/dino")
def dino():
    return FileResponse(DINO_HTML, media_type="text/html")


@app.post("/predict")
def predict(payload: dict):
    return router.predict(payload["state"], payload["questions"])
