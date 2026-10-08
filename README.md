# Predict API

A sample FastAPI app that serves a `/predict` endpoint returning `"hello"`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`. Interactive docs: `http://127.0.0.1:8000/docs`.

Dino Run (same server): `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/dino`.

## Test with curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "body": "I was double-charged, please refund me today or I will cancel."
    },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
          "billing": "invoices, payments, refunds",
          "technical": "bugs, outages, system errors",
          "sales": "pricing, new contracts",
          "other": "everything else"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
      },
      "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
      }
    }
  }'
```

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"state": {"body": "test message"}, "questions": {"urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent", "soon", "critical"]}}}' \
  | jq
```

Expected response:

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "urgency": {
      "type": "score",
      "score": 0.3401,
      "legend": {
        "0": "not urgent",
        "1": "soon",
        "2": "critical"
      },
      "probabilities": {
        "0": 0.6805,
        "1": 0.2989,
        "2": 0.0206
      },
      "confidence": 0.3602,
      "action": {
        "act_probability": 1.0
      }
    }
  },
  "usage": {
    "input_tokens": 33,
    "output_tokens": 0
  },
  "routing": {
    "model": "english",
    "repo": "convaiinnovations/laya",
    "reason": "English Latin text",
    "detection": {
      "script": "latin",
      "script_profile": {
        "latin": 1.0
      },
      "language": null,
      "is_english": true,
      "non_latin_fraction": 0.0
    },
    "workflow": null
  }
}
```
