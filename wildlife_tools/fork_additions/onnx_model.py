import numpy as np
import torch
import onnxruntime as ort


class OnnxReIDModel:
    """PtReIDModel-compatible wrapper around an ONNX Runtime session.

    Duck-types PtReIDModel.forward()'s (return_features, return_logits) contract so it
    drops directly into test_metrics()/test_classification() in place of the .pth model.
    """

    def __init__(self, onnx_path, device="cpu"):
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self._input_np_dtype = np.float16 if "float16" in self.session.get_inputs()[0].type else np.float32

        self.return_features = True
        self.return_logits = True

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        x_np = x.detach().cpu().numpy().astype(self._input_np_dtype)
        reduced_features, logits = self.session.run(["output", "logits"], {self._input_name: x_np})
        reduced_features = torch.from_numpy(reduced_features).float()
        logits = torch.from_numpy(logits).float()

        if self.return_features and self.return_logits:
            return reduced_features, logits
        elif self.return_features:
            return reduced_features
        elif self.return_logits:
            return logits
