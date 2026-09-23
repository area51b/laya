from fastapi import FastAPI
import laya, os
os.environ["USE_TF"] = "0"

app = FastAPI()
router = laya.Router(preload=True, device="cpu")  # for mac, use "cpu" or "cuda" on the NVIDIA GPU

@app.post("/predict")
def predict(payload: dict):
    return router.predict(payload["state"], payload["questions"])
