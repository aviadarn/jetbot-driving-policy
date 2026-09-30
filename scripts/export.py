"""Export a trained steering net to ONNX for TensorRT 8.2 on JetPack 4.6 (the Nano).

Opset 11, static 1x3x224x224, input = BGR/255 exactly as the camera delivers it
(normalisation is inside the model). dynamo=False: torch >= 2.9 defaults to the dynamo
exporter, which cannot target opset 11.

  python scripts/export.py --run runs/r3_ratio_s0 --out results/onnx/policy.onnx
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jdp.model import SteeringNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results/onnx/policy.onnx")
    ap.add_argument("--opset", type=int, default=11)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    model = SteeringNet(pretrained=False)
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location="cpu"))
    model.eval()
    dummy = torch.rand(1, 3, 224, 224)
    torch.onnx.export(model, dummy, a.out, opset_version=a.opset, input_names=["bgr"],
                      output_names=["xy"], do_constant_folding=True, dynamo=False)

    import onnx
    m = onnx.load(a.out)
    onnx.checker.check_model(m)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB, opset {a.opset}); "
          f"ops: {sorted({n.op_type for n in m.graph.node})}")

    # Parity check against PyTorch with onnxruntime if present, and save a reference
    # input/output pair so the Nano can verify its engine gives the same numbers.
    ref_in = torch.rand(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref_out = model(ref_in).numpy()
    base = os.path.splitext(a.out)[0]
    ref_in.numpy().astype(np.float32).tofile(base + "_ref_in.raw")
    ref_out.astype(np.float32).tofile(base + "_ref_out.raw")
    try:
        import onnxruntime as ort
        got = ort.InferenceSession(a.out).run(None, {"bgr": ref_in.numpy()})[0]
        print(f"onnxruntime max |diff| vs torch: {np.abs(got - ref_out).max():.2e}")
    except ImportError:
        print("onnxruntime not installed; parity checked on the Nano instead")


if __name__ == "__main__":
    main()
