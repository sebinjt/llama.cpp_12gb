"""Append a DFlash draft cache initialized from the final 2K target features."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from init_mtp_cache import ROOT, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draft", type=Path, default=ROOT / ".." / "Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf")
    ap.add_argument("--exe", type=Path, default=ROOT / "../build-cuda/bin/dflash_prefill.exe")
    a = ap.parse_args()
    state, out, draft = a.state.resolve(), a.out.resolve(), a.draft.resolve()
    m = json.loads(Path(str(state) + ".json").read_text())
    if m["kind"] != "target-only" or digest(state) != m["snapshot_sha256"]:
        raise RuntimeError("need a verified target-only snapshot")
    for feature in m["dflash_features"]:
        if digest(Path(feature["path"])) != feature["sha256"]:
            raise RuntimeError("DFlash feature hash mismatch")
    appendix, partial = Path(str(out) + ".dflash"), Path(str(out) + ".partial")
    if any(p.exists() for p in (out, appendix, partial, Path(str(appendix) + ".partial"))):
        raise RuntimeError("choose a new output path")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(a.exe.resolve()), "--model", m["model"],
           "--draft", str(draft), "--features", m["dflash_features_prefix"], "--out", str(appendix), "--ctx", str(m["ctx"])]
    start = time.monotonic()
    with Path(str(out) + ".log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with partial.open("wb") as dst, state.open("rb") as src, appendix.open("rb") as app:
        shutil.copyfileobj(src, dst, 16 << 20); shutil.copyfileobj(app, dst, 16 << 20)
    partial.rename(out)
    m.update(kind="target+dflash", snapshot=str(out), snapshot_sha256=digest(out), snapshot_bytes=out.stat().st_size,
             source_target=str(state), draft_model=str(draft), draft_model_sha256=digest(draft),
             dflash_initialization_seconds=time.monotonic() - start)
    Path(str(out) + ".json").write_text(json.dumps(m, indent=2))
    print(f"READY {out} initialization+save={m['dflash_initialization_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
