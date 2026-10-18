"""Build a handoff without labels, caches or training code."""
import argparse
import json
from pathlib import Path
import shutil


def package(destination, checkpoint, config):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new destination to preserve the existing handoff")
    code_files = ["__init__.py", "model.py", "video.py", "inference.py", "evaluation.py", "rules.py", "rules.json"]
    requirements = ["requirements-inference.txt", "requirements-windows-cu128.lock.txt", "THIRD_PARTY_NOTICES.md"]
    # Fail before creating a partial bundle when an input is missing.
    for path in [Path(checkpoint), Path(config), *(root / "cashwatch" / name for name in code_files),
                 *(root / name for name in requirements)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    (destination / "cashwatch").mkdir(parents=True)
    (destination / "weights").mkdir()
    for name in code_files:
        shutil.copyfile(root / "cashwatch" / name, destination / "cashwatch" / name)
    shutil.copyfile(checkpoint, destination / "weights" / "detector.pt")
    shutil.copyfile(config, destination / "weights" / "config.json")
    for name in requirements:
        shutil.copyfile(root / name, destination / name)
    shutil.copytree(root / "licenses", destination / "licenses")
    (destination / "tools").mkdir()
    for name in ["benchmark_http.py", "os_offline_worker.py", "verify_os_offline.ps1"]:
        shutil.copyfile(root / "tools" / name, destination / "tools" / name)
    (destination / "HANDOFF.md").write_text('''# Standalone local inference

Use Python 3.12. This package contains the model and rule catalog, without the web
application, training data, Excel, or source-video annotations.

From this directory on Windows:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements-inference.txt
.venv/Scripts/python.exe predict.py "C:/local-data/video.mov" --output result.json
```

For CPU replace the PyTorch index with https://download.pytorch.org/whl/cpu.
Inference uses local weights and performs no network requests. Result JSON
contains original JPEG frames; keep it local. Action descriptions are class
templates and legal references are conditional. This experimental all-data refit
has no established accuracy on new videos or new camera angles.

The Windows CUDA 12.8 lock is an environment record. The root application scripts
and backend are intentionally absent from this standalone package.
''', encoding="utf-8")
    (destination / "predict.py").write_text('''import argparse
import json
from pathlib import Path
from cashwatch.inference import analyze_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--output", default="result.json")
    args = parser.parse_args()
    result = analyze_video(args.video)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"result_saved": args.output, "events": len(result["violations"])}))
''', encoding="utf-8")
    print(json.dumps({"inference_bundle": str(destination), "contains_training_data": False}))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    default_checkpoint, default_config = next(
        ((root / "weights" / checkpoint, root / "weights" / config)
         for checkpoint, config in (
             ("event_localizer_detector.pt", "event_localizer_config.json"),
             ("hard_negative_detector.pt", "hard_negative_config.json"),
             ("reviewed_detector.pt", "reviewed_config.json"),
             ("detector.pt", "config.json"),
         ) if (root / "weights" / checkpoint).is_file()),
        (root / "weights" / "event_localizer_detector.pt", root / "weights" / "event_localizer_config.json"),
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/handoff_selected")
    parser.add_argument("--checkpoint", default=default_checkpoint)
    parser.add_argument("--config", default=default_config)
    args = parser.parse_args()
    package(args.output, args.checkpoint, args.config)
