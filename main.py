from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import laya, os
os.environ["USE_TF"] = "0"

app = FastAPI()
router = laya.Router(preload=True, device="cpu")  # for mac, use "cpu" or "cuda" on the NVIDIA GPU

BASE_DIR = Path(__file__).resolve().parent
DINO_HTML = BASE_DIR / "dino.html"

# Keep these in sync with AI_JUMP_LEAD_SECONDS / AI_DUCK_LEAD_SECONDS in dino.html.
JUMP_LEAD_SECONDS = 0.30
DUCK_LEAD_SECONDS = 0.35
JUMP_LATE_PIXELS = 12      # a jump is still useful if the obstacle is at most this far inside the dino
DINO_STAND_WIDTH = 44      # an obstacle further behind than this has passed the dino

ACTIONS = ("run", "jump", "duck")


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


# ---------------------------------------------------------------------------
# Decision helpers. All the arithmetic happens here, not in the model.
# ---------------------------------------------------------------------------

def required_action(obstacle: DinoObstacle) -> Literal["jump", "duck", "ignore"]:
    """The action that gets past this obstacle, or "ignore" if running is already safe."""
    if obstacle.kind == "cactus":
        return "jump"
    if obstacle.variant == "duck":
        return "duck"
    if obstacle.variant == "jump":
        return "jump"
    return "ignore"  # high bird


def find_threat(snapshot: DinoSnapshot) -> Optional[dict]:
    """Nearest obstacle that needs a reaction, with timing as it will be when the decision lands."""
    latency_seconds = snapshot.decision_latency_ms / 1000
    for obstacle in sorted(snapshot.obstacles, key=lambda o: o.x):
        need = required_action(obstacle)
        if need == "ignore":
            continue

        distance_on_arrival = obstacle.x - snapshot.speed * latency_seconds
        if distance_on_arrival + obstacle.w < -DINO_STAND_WIDTH:
            continue  # will already be behind the dino

        time_on_arrival = max(0.0, distance_on_arrival) / snapshot.speed
        if need == "jump":
            if distance_on_arrival < -JUMP_LATE_PIXELS:
                window = "missed"
            elif time_on_arrival <= JUMP_LEAD_SECONDS:
                window = "now"
            else:
                window = "early"
        else:
            reached = distance_on_arrival < 0
            window = "now" if reached or time_on_arrival <= DUCK_LEAD_SECONDS else "early"

        return {
            "kind": obstacle.kind,
            "variant": obstacle.variant,
            "need": need,
            "window": window,
            "distance_on_arrival": round(distance_on_arrival, 1),
            "time_on_arrival": round(time_on_arrival, 3),
        }
    return None


def allowed_actions(snapshot: DinoSnapshot, threat: Optional[dict]) -> set[str]:
    """Run is always valid. Jump/duck only when grounded, the right obstacle is in its window."""
    allowed = {"run"}
    if snapshot.grounded and threat and threat["window"] == "now":
        allowed.add(threat["need"])
    return allowed


def describe_state(snapshot: DinoSnapshot, threat: Optional[dict]) -> str:
    # The dino's current posture is deliberately left out: a small classifier tends to
    # echo it ("ducking" -> "duck") instead of reasoning about the obstacle.
    lines = [
        "The dino is on the ground."
        if snapshot.grounded
        else "The dino is in the air and cannot jump or duck."
    ]

    if threat is None:
        lines.append("No obstacle needs a reaction right now.")
        return " ".join(lines)

    if threat["kind"] == "cactus":
        label, how = "a cactus", "It must be jumped over."
    elif threat["need"] == "duck":
        label, how = "a head-height bird", "It must be ducked under."
    else:
        label, how = "a low bird", "It must be jumped over."
    lines.append(f"Nearest obstacle: {label}. {how}")

    if threat["distance_on_arrival"] <= 0:
        lines.append("When your decision is applied it will already be at the dino.")
    else:
        lines.append(
            f"When your decision is applied it will be {threat['time_on_arrival']:.2f} "
            "seconds from the dino."
        )

    window_text = {
        "now": "NOW, act on this obstacle immediately.",
        "early": "NOT YET, the obstacle is still too far away, so keep running.",
        "missed": "MISSED, it is too late to jump, so keep running.",
    }[threat["window"]]
    lines.append(f"Reaction window: {window_text}")
    return " ".join(lines)


def pick_action(allowed: set[str], model_choice, probabilities) -> tuple[str, str]:
    """Apply Laya's choice if it is valid, otherwise the most likely valid action."""
    if isinstance(model_choice, str) and model_choice in allowed:
        return model_choice, "Laya's choice was valid and applied."

    if isinstance(probabilities, dict):
        scored = [
            (float(p), action)
            for action, p in probabilities.items()
            if action in allowed and isinstance(p, (int, float))
        ]
        if scored:
            best = max(scored)[1]
            return best, (
                f"Laya chose {model_choice!r}, which is not valid right now "
                f"(allowed: {sorted(allowed)}); used its best valid action {best!r}."
            )

    return "run", (
        f"Laya chose {model_choice!r}, which is not valid right now "
        f"(allowed: {sorted(allowed)}); defaulted to run."
    )


@app.post("/dino/act")
def dino_act(snapshot: DinoSnapshot):
    threat = find_threat(snapshot)
    allowed = allowed_actions(snapshot, threat)
    state = describe_state(snapshot, threat)

    result = router.predict(
        {"body": state},
        {
            "action": {
                "type": "choice",
                "instructions": (
                    "Pick one action for the dino. If the reaction window is NOW and the "
                    "nearest obstacle must be jumped over, choose jump. If the reaction "
                    "window is NOW and the nearest obstacle must be ducked under, choose "
                    "duck. In every other case, including when the dino is in the air or "
                    "no obstacle needs a reaction, choose run."
                ),
                "criteria": {
                    "run": "The dino is in the air, no obstacle needs a reaction, or the reaction window is NOT YET or MISSED.",
                    "jump": "The dino is on the ground, the nearest obstacle must be jumped over, and its reaction window is NOW.",
                    "duck": "The dino is on the ground, the nearest obstacle must be ducked under, and its reaction window is NOW.",
                },
            }
        },
    )
    answers = result.get("answers", {}) if isinstance(result, dict) else {}
    answer = answers.get("action", {}) if isinstance(answers, dict) else {}
    model_choice = answer.get("choice") if isinstance(answer, dict) else None
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    confidence = answer.get("confidence") if isinstance(answer, dict) else None

    action, reason = pick_action(allowed, model_choice, probabilities)

    return {
        "action": action,
        "debug": {
            "state": state,
            "threat": threat,
            "allowed_actions": [a for a in ACTIONS if a in allowed],
            "model_choice": model_choice,
            "applied_action": action,
            "reason": reason,
            "probabilities": probabilities,
            "confidence": confidence,
        },
    }
