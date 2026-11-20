"""TruePitch API. Run from the project root:  uvicorn backend.app:app --port 8000"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from truepitch.evaluation.metrics import full_metrics
from truepitch.evaluation.report import TITLES, pooled
from truepitch.inference.audio import analyze_audio, decode_audio

log = logging.getLogger("truepitch.api")
ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 1000


def create_app(cfg: dict | None = None, predictor=None) -> FastAPI:
    cfg = cfg or yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    out = Path(cfg["paths"]["output_dir"])
    app = FastAPI(title="TruePitch")
    state = {"predictor": predictor}
    lock = threading.Lock()
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "truepitch.sqlite"

    def db():
        c = sqlite3.connect(db_path)
        c.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, ts TEXT, modality TEXT, "
                  "model TEXT, label TEXT, p_sarcastic REAL, text TEXT)")
        return c

    def get_predictor():
        with lock:
            if state["predictor"] is None:
                from truepitch.inference.service import Predictor
                try:
                    state["predictor"] = Predictor(cfg)
                except FileNotFoundError as e:
                    raise HTTPException(503, str(e))
            return state["predictor"]

    @app.get("/api/status")
    def status():
        meta_p = out / "models" / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
        cache = Path(cfg["paths"]["cache_dir"])
        ids_p = cache / "ids.json"
        audio_dir = Path(cfg["paths"].get("audio_dir", ""))
        return {
            "final_models": ({n: len(list((out / "models" / n).glob("seed*.pt"))) for n in meta["models"]}
                             if meta else {}),
            "final_meta": ({k: meta[k] for k in ("status", "created", "n_train", "n_val", "cv_metrics")}
                           if meta else None),
            "experiments_trained": sum((out / "experiments" / e / "runs.csv").exists() for e in TITLES),
            "experiments_total": len(TITLES),
            "dataset": {"annotations_found": Path(cfg["paths"]["mustard_annotations"]).exists(),
                        "n_samples": len(json.loads(ids_p.read_text())) if ids_p.exists() else None,
                        "features_cached": sorted(p.stem for p in cache.glob("*.npy")) if cache.exists() else [],
                        "audio_files": len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0},
            "encoders_loaded": state["predictor"] is not None,
        }

    @app.get("/api/experiments")
    def experiments():
        rows = []
        for eid, title in TITLES.items():
            d = out / "experiments" / eid
            if not (d / "runs.csv").exists():
                rows.append({"id": eid, "title": title, "status": "NOT RUN"})
                continue
            runs, preds = pd.read_csv(d / "runs.csv"), pd.read_csv(d / "preds.csv")
            p = pooled(preds)
            pm = full_metrics(p.label, p.pred)
            pf = runs.groupby("fold").test_macro_f1.mean()
            cfgj = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
            rows.append({"id": eid, "title": title, "status": "TRAINED",
                         "pooled_macro_f1": round(pm["macro_f1"], 4), "accuracy": round(pm["accuracy"], 4),
                         "fpr": round(pm["fpr"], 4), "fnr": round(pm["fnr"], 4),
                         "fold_mean_macro_f1": round(float(pf.mean()), 4), "fold_std": round(float(pf.std()), 4),
                         "val_macro_f1": round(float(runs.val_macro_f1.mean()), 4),
                         "checkpoints": len(list((d / "ckpt").glob("*"))) if (d / "ckpt").exists() else 0,
                         "config": {k: cfgj.get(k) for k in ("kind", "seeds", "split_hash", "hyperparameters",
                                                              "runtime_s", "device")}})
        return rows

    @app.get("/api/results")
    def results():
        p = out / "results.md"
        return {"markdown": p.read_text(encoding="utf-8") if p.exists() else None,
                "figures": [f"/figures/{f.name}" for f in sorted((out / "figures").glob("*.png"))]
                if (out / "figures").exists() else []}

    @app.post("/api/analyze_audio")
    async def analyze(audio: UploadFile = File(...)):
        try:
            return analyze_audio(decode_audio(await audio.read(), audio.filename))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/predict")
    async def predict(text: str | None = Form(None), audio: UploadFile | None = File(None),
                      use_whisper: bool = Form(False)):
        text = (text or "").strip() or None
        if text and len(text) > MAX_TEXT:
            raise HTTPException(400, f"Text longer than {MAX_TEXT} characters.")
        wav = None
        if audio is not None and audio.filename:
            try:
                wav = decode_audio(await audio.read(), audio.filename)
            except ValueError as e:
                raise HTTPException(400, str(e))
        if text is None and wav is None:
            raise HTTPException(400, "Provide a sentence and/or an audio file.")
        pr = get_predictor()
        source = "user"
        if text is None and use_whisper:
            text, source = pr.transcribe(wav) or None, "whisper (automatic, may contain errors)"
        res = pr.predict(text, wav, source)
        if res.get("transcript") and source != "user":
            res["warnings"].append("Transcript was generated by Whisper and may contain errors.")
        with db() as c:
            c.execute("INSERT INTO predictions (ts, modality, model, label, p_sarcastic, text) VALUES (?,?,?,?,?,?)",
                      (time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(res["modality"]), res["model"]["name"],
                       res["label"], res["p_sarcastic"], (text or "")[:500]))
        return res

    @app.get("/api/history")
    def history(limit: int = 20):
        with db() as c:
            rows = c.execute("SELECT ts, model, label, p_sarcastic, text FROM predictions ORDER BY id DESC LIMIT ?",
                             (max(1, min(limit, 100)),)).fetchall()
        return [dict(zip(("ts", "model", "label", "p_sarcastic", "text"), r)) for r in rows]

    if (out / "figures").exists():
        app.mount("/figures", StaticFiles(directory=out / "figures"), name="figures")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "frontend" / "index.html")

    return app


app = create_app() if (ROOT / "configs" / "default.yaml").exists() else None
