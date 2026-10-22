"""Produce a reusable target KV/recurrent-state cache with GPU layer-group prefill."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / ".." / "build-cuda" / "bin" / "group_prefill.exe"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / ".." / "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
    ap.add_argument("--exe", type=Path, default=EXE)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ctx", type=int, default=204800)
    ap.add_argument("--ubatch", type=int, default=2048)
    ap.add_argument("--split", type=int, default=32)
    ap.add_argument("--save-nextn", action="store_true", help="also retain target hidden states for later MTP initialization")
    ap.add_argument("--save-dflash", action="store_true", help="retain the final 2K target features for the local DFlash model")
    ap.add_argument("--tokens", type=int, default=0, help="cache this many tokens; default: prompt length minus one")
    args = ap.parse_args()
    model, prompt, output = args.model.resolve(), args.prompt.resolve(), args.out.resolve()
    exe = args.exe.resolve()
    if not exe.is_file() or not model.is_file() or not prompt.is_file():
        ap.error("executable, model and prompt must exist; build with tests/build_group_prefill.bat")
    if output.exists() or Path(str(output) + ".partial").exists():
        ap.error("output or its partial file already exists; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = Path(str(output) + ".log")
    command = [str(exe), "--model", str(model), "--prompt", str(prompt), "--out", str(output),
               "--ctx", str(args.ctx), "--ubatch", str(args.ubatch), "--split", str(args.split)]
    if args.tokens:
        command.extend(["--tokens", str(args.tokens)])
    nextn = Path(str(output) + ".nextn.f32")
    if args.save_nextn:
        command.extend(["--nextn-out", str(nextn)])
    if args.save_dflash:
        command.extend(["--dflash-features", str(output) + ".dflash"])
    lines, peaks = [], []
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as lf:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as proc:
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                lf.write(line + "\n")
                lf.flush()
                if line.startswith(("BEGIN ", "PROGRESS ", "SWAP ", "PREFILL_DONE ", "SAVED ", "HIDDEN_SAVED ", "COMPLETE ", "ERROR ")):
                    print(line, flush=True)
                    lines.append(line)
                    if line.startswith("PROGRESS "):
                        try:
                            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                    capture_output=True, text=True, timeout=5,
                                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                            if result.returncode == 0:
                                peaks.append(int(result.stdout.splitlines()[0]))
                        except (OSError, ValueError, subprocess.TimeoutExpired):
                            pass
            rc = proc.wait()
    if rc or not output.is_file():
        print(f"Prefill failed (exit {rc}); see {log}", file=sys.stderr)
        return rc or 1
    elapsed = time.monotonic() - start
    begin = next(line for line in lines if line.startswith("BEGIN "))
    count = int(re.search(r"cached_tokens=(\d+)", begin).group(1))
    with prompt.open("rb") as f:
        prompt_hash = hashlib.file_digest(f, "sha256").hexdigest()
    with output.open("rb") as f:
        snapshot_hash = hashlib.file_digest(f, "sha256").hexdigest()
    stat = model.stat()
    manifest = {"format": 1, "kind": "target-only", "type_k": "q4_0", "type_v": "q4_0", "boundary_type": "f32",
                "model": str(model), "model_size": stat.st_size,
                "model_mtime_ns": stat.st_mtime_ns, "prompt": str(prompt), "prompt_sha256": prompt_hash,
                "snapshot": str(output), "snapshot_sha256": snapshot_hash, "snapshot_bytes": output.stat().st_size,
                "ctx": args.ctx, "cached_tokens": count, "ubatch": args.ubatch, "split": args.split,
                "wall_seconds_including_save": elapsed, "effective_tokens_per_second": count / elapsed,
                "peak_gpu_used_mib_sampled": max(peaks) if peaks else None, "summary": lines}
    for name in ("group_prefill.exe", "llama.dll", "ggml-cuda.dll"):
        with (exe.parent / name).open("rb") as f:
            manifest[name + "_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    if args.save_nextn:
        with nextn.open("rb") as f:
            magic, rows, width, dtype = struct.unpack("<8sQII", f.read(24))
            if magic != b"QNEXTH01" or rows != count or dtype != 0 or nextn.stat().st_size != 24 + rows*width*4:
                raise RuntimeError("target hidden-state header or size does not match the snapshot")
            f.seek(0)
            manifest["target_hidden_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        manifest["target_hidden"] = str(nextn)
        manifest["target_hidden_bytes"] = nextn.stat().st_size
        manifest["target_hidden_width"] = width
    if args.save_dflash:
        manifest["dflash_features_prefix"] = str(output) + ".dflash"
        manifest["dflash_features"] = []
        for il in (6, 20, 34, 48, 62):
            path = Path(str(output) + f".dflash.layer{il}.f32")
            with path.open("rb") as f:
                magic, rows, begin, width, layer = struct.unpack("<8sQQII", f.read(32))
                if magic != b"QDFEAT01" or rows != min(2048, count) or begin != count-rows or width != 5120 or layer != il or path.stat().st_size != 32+rows*width*4:
                    raise RuntimeError("invalid DFlash target features")
                f.seek(0)
                manifest["dflash_features"].append({"path": str(path), "sha256": hashlib.file_digest(f, "sha256").hexdigest()})
    Path(str(output) + ".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"READY {output} ({count} cached tokens, {elapsed:.1f}s including save)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
