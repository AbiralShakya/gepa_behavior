# gepa_behavior

R&D for GEPA-style language-guided large behavior models in robot simulation.

## Hardware recommendations
- CPU: 8+ cores recommended (Ryzen 7 / Intel i7 or better)
- RAM: 32 GB recommended for vision + multi-env; 16 GB minimum for state-only
- GPU (training/vision/transformers): NVIDIA RTX 3060 (12GB) minimum, RTX 4090 / A6000 recommended; CUDA 12.x
- Disk: 20+ GB free for logs/checkpoints
- OS: Linux or macOS; for GPU training, Linux with recent NVIDIA driver is recommended

## Backends
- PyBullet (default): fast, lightweight; supports headless + optional RGB camera
- RoboSuite (optional): high-fidelity Mujoco-based manipulation tasks

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: RoboSuite
pip install -r requirements-robosuite.txt
# editable install
pip install -e .
```

## Run (PyBullet)
```bash
python -m gepa.experiment.runner run --config configs/default.yaml --episodes 2 --steps 200 --gepa-iters 2
```

## Run (RoboSuite)
```bash
python -m gepa.experiment.runner run --backend robosuite --episodes 2 --steps 200
```

## Notes
- Use `--prompt` to override the prompt; `--camera` to enable PyBullet RGB frames.
- For GPU training, ensure CUDA and PyTorch with CUDA are installed.