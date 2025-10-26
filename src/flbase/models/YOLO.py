import os
import torch
import torch.nn as nn
from typing import Dict, Any

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YOLOv5(nn.Module):
    """Light wrapper around the ultralytics YOLO model to expose a
    torch.nn.Module-like interface and utility methods used by FedNH.

    client_config keys used:
      - model_path or weights: path to pretrained weights or a model yaml
      - device: optional torch device string
    """

    def __init__(self, client_config: Dict[str, Any]):
        super().__init__()
        if YOLO is None:
            raise RuntimeError("ultralytics package not found. Install with `pip install ultralytics`")

        model_path = client_config.get('weights') or client_config.get('model_path') or client_config.get('model')
        # default to yolov5s pretrained if nothing provided (user must have downloaded it or ultralytics will auto-fetch)
        if model_path is None:
            model_path = 'yolov5s.pt'

        # instantiate ultralytics YOLO and keep the underlying PyTorch module
        # YOLO(...) returns a wrapper; .model is the internal torch.nn.Module
        yolo = YOLO(model_path)
        self.yolo = yolo
        self.model = yolo.model

        # move to device if provided
        device = client_config.get('device')
        if device is not None:
            try:
                self.to(device)
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the underlying PyTorch model.
        Note: ultralytics YOLO wrapper performs its own preprocessing when
        using the high-level API; here we expect x already prepared as a
        tensor batch (C, H, W) or (N, C, H, W) depending on caller.
        """
        return self.model(x)

    # Convenience wrappers used by federated training code
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def set_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        self.model.load_state_dict(state_dict, strict=strict)

    # Backwards-compatible names used in FedNH code
    def get_params(self) -> Dict[str, torch.Tensor]:
        return self.get_state_dict()

    def set_params(self, state_dict: Dict[str, torch.Tensor], exclude_keys: set = None):
        """Set params; if exclude_keys provided, skip those keys."""
        if exclude_keys is None or len(exclude_keys) == 0:
            self.set_state_dict(state_dict, strict=False)
            return
        # filter out excluded keys
        filtered = {k: v for k, v in state_dict.items() if k not in exclude_keys}
        # load existing state and update
        cur = self.model.state_dict()
        cur.update(filtered)
        self.set_state_dict(cur, strict=False)

    def save_weights(self, path: str):
        """Save ultralytics model weights (saves the underlying model state dict as a .pt)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: str):
        """Load state dict into the model from a .pt file."""
        state = torch.load(path, map_location='cpu')
        # if the file stores a full ultralytics checkpoint, try to extract 'model'
        if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
            state = state['model']
        self.set_state_dict(state, strict=False)

        