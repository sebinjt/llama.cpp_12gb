"""Add a fully initialized MTP draft state to an offline target snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ubatch", type=int, default=1024)
    a = ap.parse_args()
    state, out = a.state.resolve(), a.out.resolve()
    manifest = json.loads(Path(str(state) + ".json").read_text())
    hidden = Path(manifest["target_hidden"])
    model, prompt = Path(manifest["model"]), Path(manifest["prompt"])
    st = model.stat()
    if manifest["kind"] != "target-only" or st.st_size != manifest["model_size"] or st.st_mtime_ns != manifest["model_mtime_ns"]:
        raise RuntimeError("snapshot model identity mismatch or snapshot already includes a draft")
    for path, key in ((state, "snapshot_sha256"), (hidden, "target_hidden_sha256"), (prompt, "prompt_sha256")):
        if digest(path) != manifest[key]:
            raise RuntimeError(f"snapshot provenance mismatch: {path}")
    appendix, partial = Path(str(out) + ".mtp"), Path(str(out) + ".partial")
    if any(p.exists() for p in (out, appendix, partial, Path(str(appendix) + ".partial"))):
        raise RuntimeError("choose a new output path")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ROOT / "../build-cuda/bin/mtp_prefill.exe"), "--model", str(model),
           "--prompt", str(prompt), "--hidden", str(hidden), "--out", str(appendix),
           "--ctx", str(manifest["ctx"]), "--ubatch", str(a.ubatch)]
    start = time.monotonic()
    with Path(str(out) + ".log").open("w", encoding="utf-8") as lf:
        with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as p:
            for raw in p.stdout:
                line = raw.decode(errors="replace").rstrip()
                lf.write(line + "\n"); lf.flush()
                if line.startswith(("MTP_", "ERROR")):
                    print(line, flush=True)
            if p.wait():
                raise RuntimeError(f"MTP initialization failed; see {out}.log")
    with partial.open("wb") as dst, state.open("rb") as src, appendix.open("rb") as app:
        shutil.copyfileobj(src, dst, 16 << 20)
        shutil.copyfileobj(app, dst, 16 << 20)
    partial.rename(out)
    manifest.update(kind="target+mtp", snapshot=str(out), snapshot_sha256=digest(out),
                    snapshot_bytes=out.stat().st_size, source_target=str(state),
                    mtp_initialization_seconds=time.monotonic() - start, mtp_ubatch=a.ubatch)
    Path(str(out) + ".json").write_text(json.dumps(manifest, indent=2))
    print(f"READY {out} MTP initialization+save={manifest['mtp_initialization_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
