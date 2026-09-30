"""Minimal TensorRT 8.2 runner for the Jetson Nano, Python 3.6, no pip installs.

JetPack 4.6 ships the `tensorrt` bindings but not pycuda, and cuda-python needs CUDA 11+.
Device memory and copies go through ctypes into libcudart.so.10.2 instead.
"""
import ctypes
import numpy as np
import tensorrt as trt

_cudart = ctypes.CDLL("libcudart.so.10.2")
_H2D, _D2H = 1, 2
_LOGGER = trt.Logger(trt.Logger.WARNING)


def _check(rc, what):
    if rc != 0:
        raise RuntimeError("%s failed: cudaError %d" % (what, rc))


class TRTModel(object):
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = ctypes.c_void_p()
        _check(_cudart.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host = np.empty(shape, dtype=dtype)
            dev = ctypes.c_void_p()
            _check(_cudart.cudaMalloc(ctypes.byref(dev), ctypes.c_size_t(host.nbytes)), "cudaMalloc")
            self.bindings.append(dev.value)
            slot = {"name": self.engine.get_binding_name(i), "host": host, "dev": dev}
            (self.inputs if self.engine.binding_is_input(i) else self.outputs).append(slot)

    def __call__(self, x):
        inp = self.inputs[0]
        np.copyto(inp["host"], x.reshape(inp["host"].shape).astype(inp["host"].dtype, copy=False))
        _check(_cudart.cudaMemcpyAsync(inp["dev"], inp["host"].ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_size_t(inp["host"].nbytes), _H2D, self.stream), "H2D")
        self.ctx.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.value)
        for out in self.outputs:
            _check(_cudart.cudaMemcpyAsync(out["host"].ctypes.data_as(ctypes.c_void_p), out["dev"],
                                           ctypes.c_size_t(out["host"].nbytes), _D2H, self.stream), "D2H")
        _check(_cudart.cudaStreamSynchronize(self.stream), "sync")
        return [o["host"].copy() for o in self.outputs]

    def close(self):
        for s in self.inputs + self.outputs:
            _cudart.cudaFree(s["dev"])
        _cudart.cudaStreamDestroy(self.stream)


def nms(boxes, scores, iou_thres):
    """Greedy NMS, boxes as x1,y1,x2,y2 (numpy, CPU)."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def yolo_boxes(pred, conf_thres=0.25, iou_thres=0.45):
    """YOLOv5 raw head (1, N, 5 + classes) -> list of (x1, y1, x2, y2, conf, cls).
    Thresholds are the 2024 live settings (conf 0.25, IoU 0.45)."""
    p = pred.reshape(-1, pred.shape[-1])
    cls_conf = p[:, 5:] * p[:, 4:5]
    cls = cls_conf.argmax(1)
    conf = cls_conf[np.arange(len(p)), cls]
    m = conf > conf_thres
    if not m.any():
        return []
    p, conf, cls = p[m], conf[m], cls[m]
    xy, wh = p[:, :2], p[:, 2:4] / 2
    boxes = np.concatenate([xy - wh, xy + wh], 1)
    return [tuple(boxes[i]) + (float(conf[i]), int(cls[i])) for i in nms(boxes, conf, iou_thres)]
