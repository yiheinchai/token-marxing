"""End-to-end pre-submission check for train.py. Run before every `jobq.py submit`.

Verifies the train.py contract: interface/shape, causality, parameter cap, and that a
3-second smoke training run completes and prints a finite val_bpb.
Exit code 0 = PASS, 1 = FAIL.
"""
import math
import os
import subprocess
import sys
import time
import traceback

import torch

import prepare


def main():
    failures = []
    t0 = time.time()
    print("== check 1/4: import + build_model()")
    try:
        import train
        model = train.build_model()
        print(f"   model class: {type(model).__name__}")
    except Exception:
        traceback.print_exc()
        print("CHECK FAILED: could not import train.py / build model")
        return 1

    print("== check 2/4: parameter count")
    n = prepare.count_params(model)
    print(f"   params = {n:,} (cap {prepare.MAX_PARAMS:,})")
    if n > prepare.MAX_PARAMS:
        failures.append(f"too many parameters: {n:,} > {prepare.MAX_PARAMS:,}")

    print("== check 3/4: shapes + causality")
    try:
        model.eval()
        x = torch.randint(0, 256, (2, prepare.SEQ_LEN))
        with torch.no_grad():
            out = model(x)
        print(f"   input {tuple(x.shape)} -> logits {tuple(out.shape)} dtype={out.dtype}")
        if tuple(out.shape) != (2, prepare.SEQ_LEN, prepare.VOCAB_SIZE):
            failures.append(f"bad logits shape {tuple(out.shape)}")
        else:
            t = prepare.SEQ_LEN // 2
            x2 = x.clone()
            x2[:, t + 1 :] = torch.randint(0, 256, x2[:, t + 1 :].shape)
            with torch.no_grad():
                out2 = model(x2)
            diff = (out[:, : t + 1] - out2[:, : t + 1]).abs().max().item()
            print(f"   max |delta logits[:<=t]| after perturbing future = {diff:.2e}")
            if diff > 1e-4:
                failures.append(f"model is not causal (future leak, delta={diff:.2e})")
            if not torch.isfinite(out).all():
                failures.append("non-finite logits")
    except Exception:
        traceback.print_exc()
        failures.append("forward pass raised")

    print("== check 4/4: 3s smoke training run")
    env = dict(os.environ, SMOKE="1")
    proc = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, env=env,
                          cwd=os.path.dirname(os.path.abspath(__file__)), timeout=300)
    for line in proc.stdout.splitlines():
        print("   | " + line)
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        failures.append(f"smoke run exited with code {proc.returncode}")
    else:
        vals = [l for l in proc.stdout.splitlines() if l.startswith("val_bpb:")]
        if not vals:
            failures.append("smoke run did not print val_bpb")
        elif not math.isfinite(float(vals[-1].split(":")[1])):
            failures.append("smoke val_bpb is not finite")

    print(f"== elapsed {time.time() - t0:.1f}s")
    if failures:
        for f in failures:
            print("CHECK FAILED: " + f)
        return 1
    print("CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
