from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import laya, os
os.environ["USE_TF"] = "0"

app = FastAPI()
router = laya.Router(preload=True, device="cpu")  # for mac, use "cpu" or "cuda" on the NVIDIA GPU

BASE_DIR = Path(__file__).resolve().parent
DINO_HTML = BASE_DIR / "dino.html"


class DinoObstacle(BaseModel):
    kind: Literal["cactus", "bird"]
    variant: str
    x: float = Field(allow_inf_nan=False)
    y: float = Field(ge=0, allow_inf_nan=False)
    w: float = Field(gt=0, allow_inf_nan=False)
    h: float = Field(gt=0, allow_inf_nan=False)


class DinoSnapshot(BaseModel):
    grounded: bool
    ducking: bool
    speed: float = Field(gt=0, allow_inf_nan=False)
    score: float = Field(ge=0, allow_inf_nan=False)
    decision_latency_ms: float = Field(default=0, ge=0, le=5000, allow_inf_nan=False)
    obstacles: list[DinoObstacle] = Field(default_factory=list, max_length=2)


@app.get("/")
@app.get("/dino")
def dino():
    return FileResponse(DINO_HTML, media_type="text/html")


@app.post("/predict")
def predict(payload: dict):
    return router.predict(payload["state"], payload["questions"])


@app.post("/dino/act")
def dino_act(snapshot: DinoSnapshot):
    posture = "ducking" if snapshot.ducking else "grounded" if snapshot.grounded else "airborne"
    details = [
        f"The dino is {posture}.",
        f"Speed is {snapshot.speed:.0f} pixels/second and score is {snapshot.score:.0f}.",
        f"Recent end-to-end decision latency is {snapshot.decision_latency_ms:.0f} ms.",
    ]

    for index, obstacle in enumerate(snapshot.obstacles, start=1):
        contact_distance = max(0, obstacle.x)
        contact_time = contact_distance / snapshot.speed
        distance_on_arrival = (
            obstacle.x - snapshot.speed * snapshot.decision_latency_ms / 1000
        )
        time_on_arrival = max(0, distance_on_arrival) / snapshot.speed
        description = f"Obstacle {index}: {obstacle.variant} {obstacle.kind}"
        if obstacle.kind == "bird":
            if obstacle.variant == "duck":
                description += "; head-height bird, standing would hit, ducking can pass under"
            elif obstacle.variant == "jump":
                description += "; low bird, ducking would still hit, jump over it"
            elif obstacle.variant == "safe":
                description += "; high bird, safe to run under"
        else:
            description += "; cactus, jump over it"
        distance = (
            f"{obstacle.x:.0f} pixels ahead"
            if obstacle.x >= 0
            else f"{abs(obstacle.x):.0f} pixels into the dino's horizontal space"
        )
        details.append(
            f"{description}; {distance}, "
            f"{contact_time:.2f} seconds to contact, at y={obstacle.y:.0f} "
            f"with size {obstacle.w:.0f}x{obstacle.h:.0f}. "
            f"Estimated distance when your decision reaches the game: "
            f"{distance_on_arrival:.0f} pixels "
            f"({time_on_arrival:.2f} seconds before contact)."
        )

    if not snapshot.obstacles:
        details.append("There are no upcoming obstacles in view.")

    result = router.predict(
        {"body": " ".join(details)},
        {
            "action": {
                "type": "choice",
                "instructions": (
                    "Choose the safest action that should be applied after the estimated "
                    "decision latency. For a cactus or low bird, choose jump only when the "
                    "dino is grounded and the nearest such obstacle is predicted to be within "
                    "0.30 seconds of contact when the decision arrives; if it is farther away, "
                    "choose run and wait for a later decision. If the dino is airborne, choose "
                    "run because it cannot start another jump. For a head-height bird, choose "
                    "duck when it will reach the dino within 0.35 seconds, or is already "
                    "overlapping the dino. High birds are safe to run under."
                ),
                "criteria": {
                    "run": "The dino is airborne, the next jump obstacle is more than 0.30 seconds away on decision arrival, or only a safe high bird is present.",
                    "jump": "The dino is grounded and a cactus or low bird will be within 0.30 seconds of contact when this decision arrives.",
                    "duck": "A head-height bird will be within 0.35 seconds of contact on decision arrival or is already overlapping the dino.",
                },
            }
        },
    )
    answers = result.get("answers", {}) if isinstance(result, dict) else {}
    answer = answers.get("action", {}) if isinstance(answers, dict) else {}
    model_choice = answer.get("choice") if isinstance(answer, dict) else None
    valid_choice = isinstance(model_choice, str) and model_choice in {"run", "jump", "duck"}
    action = model_choice if valid_choice else "run"
    reason = (
        "Laya's choice was applied."
        if valid_choice
        else f"Unrecognized Laya choice {model_choice!r}; defaulted to run."
    )

    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    confidence = answer.get("confidence") if isinstance(answer, dict) else None
    return {
        "action": action,
        "debug": {
            "state": " ".join(details),
            "model_choice": model_choice,
            "applied_action": action,
            "reason": reason,
            "probabilities": probabilities,
            "confidence": confidence,
        },
    }
