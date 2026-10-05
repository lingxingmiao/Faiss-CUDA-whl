#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify a built faiss GPU artefact.

Checks
  1. the DLL exists and contains the required SASS architectures
     (default: sm_60 + sm_70 -- what conda-forge/PyPI builds are missing);
  2. optionally a PTX entry for forward compatibility;
  3. optionally the wheel layout: faiss/_swigfaiss*.pyd, faiss/_gpu_build.py,
     bundled deps in faiss.libs, and -- for the "full" wheel -- bundled CUDA
     runtime DLLs.
"""
import argparse
import os
import re
import subprocess
import sys
import zipfile


def log(m=""):
    print(m, flush=True)


def find_cuobjdump(explicit=None):
    if explicit:
        return explicit
    root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if root:
        for name in ("cuobjdump.exe", "cuobjdump"):
            cand = os.path.join(root, "bin", name)
            if os.path.isfile(cand):
                return cand
    return "cuobjdump"


def list_archs(cuobjdump, dll):
    out = {}
    for mode, flag in (("sass", "--list-elf"), ("ptx", "--list-ptx")):
        r = subprocess.run([cuobjdump, flag, dll], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("%s failed: %s" % (flag, r.stderr[:400]))
        exts = r"\.sm_(\d+[a-z]?)\.(?:cubin|ptx)"
        out[mode] = sorted(set(re.findall(exts, r.stdout)),
                           key=lambda a: int(re.match(r"\d+", a).group(0)))
    return out


def check_dll(args):
    dll = args.dll
    if not os.path.isfile(dll):
        log("FAIL: dll not found: %s" % dll)
        return 1
    size = os.path.getsize(dll) / (1 << 20)
    log("dll      : %s (%.1f MB)" % (dll, size))
    archs = list_archs(find_cuobjdump(args.cuobjdump), dll)
    log("sass     : %s" % ",".join(archs["sass"]))
    log("ptx      : %s" % ",".join(archs["ptx"]))

    required = [a.strip() for a in args.require.split(",") if a.strip()]
    missing = [r for r in required
               if not any(a == r or a.startswith(r) for a in archs["sass"])]
    if missing:
        log("FAIL: missing SASS for sm_%s" % ",".join(missing))
        return 1
    log("required : sm_%s present  OK" % ", sm_".join(required))

    if len(archs["sass"]) < args.min_archs:
        log("FAIL: only %d archs (expected >= %d)"
            % (len(archs["sass"]), args.min_archs))
        return 1
    if not archs["ptx"]:
        log("WARN: no PTX embedded (no forward compatibility)")
    return 0


CUDA_DLL_RE = re.compile(
    r"(cublasLt64|cublas64|cudart64|curand64|nvrtc64|nvJitLink|cufft64|"
    r"cusparse64|cusolver64|nppial64|nppidei64|nppif64|nppig64|nppim64|"
    r"nppist64|nppisu64|nppitc64|npps64)", re.I)


def check_wheel(args):
    whl = args.wheel
    if not os.path.isfile(whl):
        log("FAIL: wheel not found: %s" % whl)
        return 1
    log("")
    log("wheel    : %s (%.1f MB)" % (whl, os.path.getsize(whl) / (1 << 20)))
    with zipfile.ZipFile(whl) as z:
        names = z.namelist()
    pyd = [n for n in names if re.match(r"faiss/_swigfaiss.*\.pyd$", n)]
    marker = [n for n in names if n.endswith("faiss/_gpu_build.py")]
    # delvewheel 把 vendored DLL 放进 <发行版名>.libs/（这里是 faiss_gpu_cu129.libs），
    # 不叫固定名字 faiss.libs；按 ".libs/" 后缀识别，否则这个检查等于没看
    libs = sorted({n.split("/", 1)[1] for n in names
                   if ".libs/" in n and n.split("/", 1)[1]})
    log("pyd      : %s" % (pyd or "MISSING"))
    log("marker   : %s" % (marker or "MISSING"))
    log("bundled  : %s" % (", ".join(libs) if libs else "(none)"))
    rc = 0
    if not pyd:
        log("FAIL: no _swigfaiss*.pyd inside the wheel")
        rc = 1
    if not marker:
        log("FAIL: faiss/_gpu_build.py missing -> GPU preload will not run")
        rc = 1
    cuda_libs = [n for n in libs if CUDA_DLL_RE.search(n)]
    if args.expect_bundled_cuda:
        if not cuda_libs:
            log("FAIL: expected bundled CUDA DLLs (full wheel) but found none")
            rc = 1
        else:
            log("cuda in wheel: %s  OK" % ", ".join(cuda_libs))
    else:
        big = [n for n in cuda_libs if "cublas" in n.lower()]
        if big:
            log("FAIL: light wheel must not bundle cublas (%s)" % big)
            rc = 1
        else:
            log("light wheel carries no CUDA runtime  OK")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll")
    ap.add_argument("--wheel")
    ap.add_argument("--cuobjdump", default=None)
    ap.add_argument("--require", default="60,70")
    ap.add_argument("--min-archs", type=int, default=4)
    ap.add_argument("--expect-bundled-cuda", action="store_true")
    args = ap.parse_args()

    if not args.dll and not args.wheel:
        log("nothing to check")
        return 1
    rc = 0
    if args.dll:
        rc |= check_dll(args)
        log("")
    if args.wheel:
        rc |= check_wheel(args)
    log("")
    log("verify_artifact: %s" % ("OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
