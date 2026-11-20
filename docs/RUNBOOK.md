# Runbook (run from the project root, venv active)

1. python -m pytest -q                          # expect: 14 passed
2. python scripts/extract_features.py --only qc # audio sanity check (seconds)
3. python scripts/extract_features.py --only text      # ~3-6 min CPU, downloads ~0.8 GB
4. python scripts/extract_features.py --only egemaps   # ~5-10 min
5. python scripts/extract_features.py --only w2v       # ~8-20 min, downloads ~0.4 GB
6. python scripts/extract_features.py --only valence   # ~15-40 min, downloads ~0.7 GB
   (the script is resumable: re-running skips finished features; --force recomputes)
7. python scripts/run_experiments.py --exp E1 E2 E3 --seeds 5   # baselines
8. python scripts/run_experiments.py --exp E4 E5 E6 E7 E8 E9 --seeds 5
9. python scripts/evaluate.py                   # -> outputs/results.md, outputs/figures/*.png

Outputs per experiment: outputs/experiments/E*/{runs.csv,preds.csv,history.json,config.json,train.log,ckpt/}

# Final models and app
10. python scripts/train_final.py                    # trains the models the app serves (outputs/models)
11. uvicorn backend.app:app --port 8000               # run from the project root
    then open http://127.0.0.1:8000 in a browser. First prediction loads the encoders (about a minute on CPU).
