#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ask nvcc which architectures it supports and emit a CMake arch list.

Why: conda-forge / PyPI faiss-gpu builds ship `sm_53;62;72;75;80;86;89;90`
(PTX only for the newest), i.e. sm_60 (Tesla P100) and sm_70 (Tesla V100) are
missing and every GPU kernel launch fails with
`CUDA error 209 no kernel image is available for execution on the device`.

This script enumerates *all* archs the installed toolkit knows about so the
resulting DLL is really "全 CUDA 架构".

Modes
  major : (default) one cubin per major version, the *lowest* minor of each
          (60, 70, 80, 90, 100, 120...). SASS is minor-forward compatible
          within a major, so sm_60 covers sm_61/sm_62 and sm_70 covers
          sm_72/sm_75 -- full coverage with the fewest kernels (fast build).
  min60 : every arch >= sm_60 that the toolkit reports (slow, huge DLL).
  all   : everything nvcc knows, including ancient archs.

Output: e.g. 60-real;70-real;80-real;90-real;100-real;120-real;121-virtual
(`-virtual` on the newest one adds PTX so future GPUs can JIT.)
"""
import argparse
import os
import re
import subprocess
import sys


def find_nvcc(explicit=None):
    if explicit:
        return explicit
    nvcc = os.environ.get("CUDACXX")
    if nvcc and os.path.isfile(nvcc):
        return nvcc
    root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if root:
        for name in ("nvcc.exe", "nvcc"):
            cand = os.path.join(root, "bin", name)
            if os.path.isfile(cand):
                return cand
    return "nvcc"


def list_archs(nvcc):
    out = subprocess.run([nvcc, "--list-gpu-arch"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("nvcc --list-gpu-arch failed: %s%s"
                           % (out.stdout, out.stderr))
    archs = []
    for line in out.stdout.splitlines():
        m = re.match(r"\s*compute_(\d+[a-z]?)\s*$", line)
        if m:
            archs.append(m.group(1))
    if not archs:
        raise RuntimeError("could not parse nvcc arch list:\n%s" % out.stdout)
    return archs


def numeric(arch):
    return int(re.match(r"(\d+)", arch).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvcc", default=None)
    ap.add_argument("--min-sm", type=int, default=60,
                    help="drop archs older than this (default 60 = P100)")
    ap.add_argument("--mode", default="major", choices=["major", "min60", "all"])
    ap.add_argument("--require", default="60,70",
                    help="comma separated archs that MUST be present")
    args = ap.parse_args()

    nvcc = find_nvcc(args.nvcc)
    archs = list_archs(nvcc)
    print("nvcc      : %s" % nvcc, file=sys.stderr)
    print("all archs : %s" % ",".join(archs), file=sys.stderr)

    min_sm = 0 if args.mode == "all" else args.min_sm
    kept = [a for a in archs if numeric(a) >= min_sm]
    if not kept:
        raise RuntimeError("no arch >= sm_%d in %s" % (min_sm, archs))

    if args.mode == "major":
        # keep the lowest minor of each major version; SASS is minor-forward
        # compatible inside a major, so one cubin per major is enough.
        by_major = {}
        for a in kept:
            major = numeric(a) // 10
            if major not in by_major or numeric(a) < numeric(by_major[major]):
                by_major[major] = a
        kept = [by_major[k] for k in sorted(by_major)]

    required = [r.strip() for r in args.require.split(",") if r.strip()]
    missing = [r for r in required
               if not any(a.startswith(r) for a in kept)]
    if missing:
        raise RuntimeError(
            "toolkit %s cannot build required arch(s) sm_%s -- CUDA 13 dropped "
            "sm_60/sm_70! archs=%s" % (nvcc, ",".join(missing), kept))

    newest = max(kept, key=numeric)
    parts = ["%s-real" % a for a in kept]
    parts.append("%s-virtual" % newest)  # PTX for forward compatibility
    cmake_list = ";".join(parts)
    print("cmake      : %s" % cmake_list, file=sys.stderr)
    print(cmake_list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
