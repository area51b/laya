---
title: Dino Run with Laya
emoji: 🦖
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Dino Run with Laya

A small runner game where you can switch to **AI** mode and let the
[Laya](https://pypi.org/project/laya/) decision model choose between run, jump and duck.

- `GET /` serves the game.
- `POST /dino/act` takes a snapshot of the game and returns the action to take.
- `POST /predict` is raw model access and is **disabled by default**. Add a Space secret named
  `PREDICT_TOKEN` to enable it; callers then send that value in an `x-api-key` header.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/.

## Notes

- The model loads at startup, so the first request after the Space wakes up can be slow.
- Each AI decision is a network round trip plus inference. Turn on **Local timing assist** in the
  game if the dino reacts late.
