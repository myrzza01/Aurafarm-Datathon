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
    (destination / "cashwatch").mkdir(parents=True)
    (destination / "weights").mkdir()
    for name in ["__init__.py", "model.py", "video.py", "inference.py", "evaluation.py"]:
        shutil.copyfile(root / "cashwatch" / name, destination / "cashwatch" / name)
    shutil.copyfile(checkpoint, destination / "weights" / "detector.pt")
    shutil.copyfile(config, destination / "weights" / "config.json")
    for name in ["requirements-inference.txt", "requirements-lock.txt", "THIRD_PARTY_NOTICES.md", "HANDOFF.md"]:
        shutil.copyfile(root / name, destination / name)
    shutil.copytree(root / "licenses", destination / "licenses")
    (destination / "tools").mkdir()
    for name in ["benchmark_http.py", "os_offline_worker.py", "verify_os_offline.ps1"]:
        shutil.copyfile(root / "tools" / name, destination / "tools" / name)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/handoff_selected")
    parser.add_argument("--checkpoint", default="artifacts/cycle/selected.pt")
    parser.add_argument("--config", default="artifacts/cycle/selected_config.json")
    args = parser.parse_args()
    package(args.output, args.checkpoint, args.config)
