"""Launch an isolated interpreter with the handoff runtime and anonymous video inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/handoff_selected")
    parser.add_argument("--manifest", default="artifacts/data/manifest.json")
    parser.add_argument("--output", default="artifacts/cycle/offline_selected")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    dev = [r for r in manifest["records"] if r["split"] in {"dev", "val"}]
    source = next(r for r in dev if r["label"] == 1)
    longest = max(manifest["records"], key=lambda r: r["duration_sec"])
    first, alias, large = [output / name for name in ["input_a.mov", "completely_unknown_name.mov", "longest.mov"]]
    shutil.copyfile(source["path"], first)
    shutil.copyfile(source["path"], alias)
    shutil.copyfile(longest["path"], large)
    bundle = Path(args.bundle).resolve()
    assert not (bundle / "cashwatch" / "data.py").exists()
    assert not (bundle / "cashwatch" / "train.py").exists()
    report_path = output / "checks.json"
    command = [sys.executable, "-I", str(root / "tools" / "offline_worker.py"),
        "--bundle", str(bundle), "--video", str(first), "--alias", str(alias),
        "--longest", str(large), "--output", str(report_path)]
    started = time.monotonic()
    subprocess.run(command, cwd=output, check=True, timeout=300)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["separate_process_wall_sec"] = time.monotonic()-started
    report["longest_video_id"] = longest["id"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
