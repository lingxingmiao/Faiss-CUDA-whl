#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve which faiss version to build and emit the build matrix.

Outputs (written to $GITHUB_OUTPUT when running in Actions):
  version   : upstream faiss tag that will be built (e.g. v1.14.3)
  build     : 'true' / 'false'  (false = this version is already released here)
  matrix    : JSON array of build jobs: {cuda, cuda_short, cuda_tag, toolset,
              python, py_tag, arch_mode}

No third-party dependencies (urllib only) so the `plan` job stays trivial.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

UPSTREAM = "facebookresearch/faiss"

# CUDA -> MSVC toolset pin. CUDA < 13 is mandatory here: CUDA 13 dropped
# sm_60 (P100) and sm_70 (V100), so it cannot produce the artefacts this
# repository exists for.
CUDA_TOOLSET = {
    "11.8.0": "14.2",
    "12.1.1": "14.3",
    "12.4.1": "14.3",
    "12.6.3": "14.3",
    "12.8.1": "14.4",
    "12.9.0": "14.4",
}

DEFAULT_CUDA = "12.9.0,12.6.3"
DEFAULT_PYTHON = "3.10,3.11,3.12,3.13"


def log(msg=""):
    print(msg, flush=True)


def api(path, token=None):
    url = "https://api.github.com" + path
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "faiss-cuda-whl-ci",
    })
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def latest_upstream_tag(token):
    rel = api("/repos/%s/releases/latest" % UPSTREAM, token)
    return rel["tag_name"]


def repo_has_release(repo, tag, token):
    if not repo:
        return False
    try:
        api("/repos/%s/releases/tags/%s" % (repo, tag), token)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def cuda_tag(version):
    """12.9.0 -> cu129 (PyTorch-style tag)."""
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError("bad CUDA version: %s" % version)
    return "cu%s%s" % (parts[0], parts[1])


def build_matrix(cuda_versions, python_versions, arch_mode):
    entries = []
    for cuda in cuda_versions:
        major = int(cuda.split(".")[0])
        if major >= 13:
            log("!! skip CUDA %s: CUDA 13 dropped sm_60/sm_70 support" % cuda)
            continue
        toolset = CUDA_TOOLSET.get(cuda)
        if toolset is None:
            log("!! skip CUDA %s: no MSVC toolset pin known" % cuda)
            continue
        short = ".".join(cuda.split(".")[:2])
        for py in python_versions:
            entries.append({
                "cuda": cuda,
                "cuda_short": short,
                "cuda_tag": cuda_tag(cuda),
                "toolset": toolset,
                "python": py,
                "py_tag": "cp" + py.replace(".", ""),
                "arch_mode": arch_mode,
            })
    return entries


def emit(version, build, matrix):
    out = {
        "version": version,
        "build": "true" if build else "false",
        "matrix": json.dumps(matrix, separators=(",", ":")),
    }
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            for k, v in out.items():
                f.write("%s=%s\n" % (k, v))
    for k, v in out.items():
        log("%s = %s" % (k, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=os.environ.get("IN_VERSION", ""),
                    help="faiss tag to build; default = latest upstream release")
    ap.add_argument("--force", default=os.environ.get("IN_FORCE", "false"),
                    help="true = rebuild even if the release already exists")
    ap.add_argument("--cuda", default=os.environ.get("IN_CUDA") or DEFAULT_CUDA)
    ap.add_argument("--python", default=os.environ.get("IN_PYTHON") or DEFAULT_PYTHON)
    ap.add_argument("--arch-mode", default=os.environ.get("IN_ARCH_MODE") or "min60",
                    choices=["min60", "all"])
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    args = ap.parse_args()

    force = str(args.force).lower() in ("1", "true", "yes")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

    version = args.version.strip()
    if not version:
        version = latest_upstream_tag(token)
        log("latest upstream faiss release: %s" % version)

    if not re.match(r"^v?\d+\.\d+", version):
        log("!! refusing to build suspicious tag %r" % version)
        emit(version, False, [])
        return 1

    matrix = build_matrix(
        [c.strip() for c in args.cuda.split(",") if c.strip()],
        [p.strip() for p in args.python.split(",") if p.strip()],
        args.arch_mode,
    )
    if not matrix:
        log("!! empty build matrix")
        emit(version, False, [])
        return 1

    release_tag = "faiss-%s" % version
    if not force and repo_has_release(args.repo, release_tag, token):
        log("release %s already exists in %s -> nothing to do"
            % (release_tag, args.repo))
        emit(version, False, [])
        return 0

    log("")
    log("planned jobs (%d):" % len(matrix))
    for e in matrix:
        log("  cuda %-8s py %-5s toolset %-5s arch %s"
            % (e["cuda"], e["python"], e["toolset"], e["arch_mode"]))
    emit(version, True, matrix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
