#!/usr/bin/env python3
"""Build the ABS adapter against this baseline's separately built FAISS binary."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from abs_runtime import ROOT, DEFAULT_FAISS_SOURCE, active_binary, digest, load_faiss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faiss-python-path", type=Path)
    parser.add_argument("--faiss-source", type=Path,
                        default=Path(os.environ.get("ABS_FAISS_SOURCE", DEFAULT_FAISS_SOURCE)))
    parser.add_argument("--build-dir", type=Path,
                        default=Path(os.environ.get("ABS_BUILD_DIR", ROOT / "build")))
    args = parser.parse_args()
    faiss = load_faiss(args.faiss_python_path)
    binary = active_binary()
    headers = args.faiss_source.resolve()
    output = args.build_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "scripts/native_abs.cpp"
    inputs = [source] + [headers / name for name in (
        "faiss/Index.h", "faiss/IndexFlat.h", "faiss/IndexHNSW.h", "faiss/impl/HNSW.h",
        "faiss/impl/DistanceComputer.h", "faiss/impl/VisitedTable.h",
        "faiss/impl/VisitedTable.cpp", "faiss/IndexHNSW.cpp")]
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"Missing source/header: {path}")
    compiler = os.environ.get("CXX", "g++")
    temporary = output / "libpaper_abs.tmp.so"
    library = output / "libpaper_abs.so"
    command = [compiler, "-std=c++17", "-O3", "-Wall", "-Wextra", "-fPIC", "-shared", "-fopenmp",
               "-I" + str(headers), str(source), str(binary),
               "-Wl,-rpath," + str(binary.parent), "-o", str(temporary)]
    subprocess.run(command, check=True)
    temporary.replace(library)
    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(), "command": command,
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "paper": "arXiv:2505.15636v1, Algorithm 2", "boundary": "stop >=; admit <",
        "faiss_python_path": str(Path(faiss.__file__).resolve().parent.parent),
        "faiss_binary": str(binary), "faiss_binary_sha256": digest(binary),
        "faiss_compile_options": faiss.get_compile_options(),
        "source_sha256": {str(p): digest(p) for p in inputs},
        "library_sha256": digest(library),
    }
    (output / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {library}\nFAISS: {binary}")


if __name__ == "__main__":
    main()
