"""Checkpoint helper for the strong-baseline chains on lib613server (2026-11-17).
  copy SRC DST           copy one file; DST must not exist yet, or be byte-identical (never overwrites anything else)
  swap DIR FROM TO KEY   the single file in DIR whose name contains FROM and KEY -> the same name with FROM replaced by TO
                         (the copy is a file this chain made itself, so an older copy is replaced)
'swap' lets WaveMLP / ASKNet test CR with the model their CRD run trained (the training data are the same: training
receivers, days 1-2, same seed and split; only the test days differ), as IB-RFF stage 1 does (CR train -> CRD test).
'copy' brings the 4090's IB-RFF stage-1 (FAN) checkpoints into a fresh stage-2 work directory.
"""
import filecmp
import os
import shutil
import sys

mode = sys.argv[1]
if mode == "copy":
    src, dst = sys.argv[2], sys.argv[3]
    assert os.path.isfile(src), f"missing {src}"
    if os.path.exists(dst):
        assert filecmp.cmp(src, dst, shallow=False), f"{dst} exists and differs; not overwritten"
        print("identical copy already there:", dst)
    else:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        print("copied", src, "->", dst)
elif mode == "swap":
    d, old, new, key = sys.argv[2:6]
    hits = [f for f in os.listdir(d) if old in f and key in f]
    assert len(hits) == 1, f"expected one checkpoint with {old!r} and {key!r} in {d}: {hits}"
    src, dst = os.path.join(d, hits[0]), os.path.join(d, hits[0].replace(old, new))
    shutil.copy2(src, dst)
    print("copied", src, "->", dst)
else:
    sys.exit(f"unknown mode {mode}")
