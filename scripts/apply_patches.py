#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply .github/patches/*.patch to a faiss checkout.

Handles three cases per patch:
  * applies cleanly                  -> applied
  * already applied (reverse applies)-> skipped (upstream fixed it)
  * neither                          -> SOFT-WARN or FAIL, depending on
                                        --strict (default: warn, and record it
                                        in the job summary so the patch set can
                                        be refreshed for the new version)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def lf_copy(patch, tmpdir):
    """Return a path to the patch with LF endings.

    Windows checkouts (git core.autocrlf=true, the Git-for-Windows default and
    what the GitHub runner uses) rewrite these .patch files to CRLF.  The faiss
    sources are LF, so every context line then carries a trailing CR and *no*
    hunk matches -- which looks exactly like "upstream moved the code".
    """
    with open(patch, "rb") as f:
        data = f.read()
    if b"\r\n" not in data:
        return patch, False
    fixed = os.path.join(tmpdir, os.path.basename(patch))
    with open(fixed, "wb") as f:
        f.write(data.replace(b"\r\n", b"\n"))
    return fixed, True


def try_apply(patch, cwd):
    """Return one of: applied / already / failed.

    Tries full context first, then -C1 so a patch survives small upstream
    context drift (the surrounding lines of a hunk are the fragile part).
    """
    last_err = ""
    for ctx in ([], ["-C1"], ["-C0"]):
        r = run(["git", "apply", "--check", "-p1"] + ctx + [patch], cwd)
        if r.returncode == 0:
            r2 = run(["git", "apply", "-p1"] + ctx + [patch], cwd)
            if r2.returncode != 0:
                return "failed", r2.stderr.strip()
            return "applied" + (" (fuzzy)" if ctx else ""), ""
        last_err = (r.stderr or "").strip() or (r.stdout or "").strip()
    # reverse-applies cleanly => the change is already in the tree
    for ctx in ([], ["-C1"]):
        r3 = run(["git", "apply", "--check", "-R", "-p1"] + ctx + [patch], cwd)
        if r3.returncode == 0:
            return "already", ""
    return "failed", last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="faiss git checkout")
    ap.add_argument("--patches", default=".github/patches")
    ap.add_argument("--strict", default="false")
    args = ap.parse_args()

    strict = str(args.strict).lower() in ("1", "true", "yes")
    files = sorted(f for f in os.listdir(args.patches) if f.endswith(".patch"))
    if not files:
        print("!! no patches found in %s" % args.patches)
        return 1

    tmpdir = tempfile.mkdtemp(prefix="faiss-patches-")
    failed = []
    try:
        for name in files:
            path = os.path.abspath(os.path.join(args.patches, name))
            path, normalized = lf_copy(path, tmpdir)
            status, err = try_apply(path, args.src)
            print("%-52s %s%s" % (name, status.upper(),
                                  "  [patch was CRLF, applied as LF]"
                                  if normalized else ""))
            if status == "failed":
                failed.append((name, err))
                for line in err.splitlines():
                    print("    %s" % line)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if failed:
        print("")
        print("!! %d patch(es) did not apply." % len(failed))
        print("   Upstream may have fixed/moved the code. Refresh "
              ".github/patches for this faiss version.")
        if strict:
            return 1
        print("   continuing because --strict=false (build may fail later)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
