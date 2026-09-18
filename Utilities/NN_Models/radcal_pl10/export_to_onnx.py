#!/usr/bin/env python3
"""Export the RadCal PL10 PyTorch checkpoint to an FDS deployment ONNX model.

Inputs are physical float32 values in the order recorded in the exported
manifest.  The full RadCal checkpoint uses:
temperature_K, x_co2, x_h2o, x_co, x_c2h4, soot_fv.
The graph embeds the normalization used during training. Its output is direct
RadCal kappa_10cm in cm^-1; FDS must multiply it by 100 for KAPPA_GAS in m^-1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

# ``model_state`` checkpoints use the feature order from train_kappa_10cm.py.
MODEL_STATE_INPUT_NAMES = ("temperature_K", "x_co2", "x_h2o", "x_co", "soot_fv", "x_c2h4")
# ``nn_radcal_model_full.pt`` was saved as a complete Python model and has a
# different final-pair order.  Its normalization statistics and accompanying
# metadata establish this order.
FULL_MODEL_INPUT_NAMES = ("temperature_K", "x_co2", "x_h2o", "x_co", "x_c2h4", "soot_fv")


class KappaNet(nn.Module):
    """Architecture used by train_kappa_10cm.py."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DeploymentModel(nn.Module):
    """Recreates the training preprocessing plus network for ONNX export."""

    def __init__(self, checkpoint: dict[str, object]) -> None:
        super().__init__()
        self.network = KappaNet()
        self.network.load_state_dict(checkpoint["model_state"])
        self.register_buffer("temperature_min", torch.tensor(float(checkpoint["temperature_min"])))
        self.register_buffer("temperature_max", torch.tensor(float(checkpoint["temperature_max"])))
        self.register_buffer("species_min", torch.as_tensor(checkpoint["species_min"], dtype=torch.float32))
        self.register_buffer("species_max", torch.as_tensor(checkpoint["species_max"], dtype=torch.float32))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        temperature = (raw[:, 0:1] - self.temperature_min) / (self.temperature_max - self.temperature_min)
        species = raw[:, 1:]
        species = torch.log1p(species / self.species_min) / torch.log1p(self.species_max / self.species_min)
        return self.network(torch.cat((temperature, species), dim=1))


class LegacyNormalization(nn.Module):
    """Z-score normalization stored in the original PL10 checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(6))
        self.register_buffer("variance", torch.ones(6))
        self.register_buffer("std", torch.ones(6))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std


class LegacyPL10Model(nn.Module):
    """Architecture encoded by the original model_state_dict checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.normalization = LegacyNormalization()
        self.fc1 = nn.Linear(6, 32)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(self.normalization(raw))))


class NormalizationLayer(nn.Module):
    """Compatibility definition for the pickled full RadCal model.

    ``nn_radcal_model_full.pt`` was saved with classes defined in ``__main__``.
    PyTorch needs matching definitions available when it unpickles that file.
    """

    def __init__(self, mean: torch.Tensor | None = None, variance: torch.Tensor | None = None) -> None:
        super().__init__()
        if mean is not None:
            self.register_buffer("mean", mean)
        if variance is not None:
            self.register_buffer("variance", variance)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # The serialized object includes ``std`` (and an ``epsilon`` retained
        # from training); its documented preprocessing is exactly z-score.
        return (values - self.mean) / self.std


class RegressionModel(nn.Module):
    """Compatibility definition for the pickled full RadCal model."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        activation = getattr(self, "activation", None) or torch.relu
        output = self.fc2(activation(self.fc1(self.normalization(raw))))
        output_activation = getattr(self, "output_activation", None)
        return output if output_activation is None else output_activation(output)


def load_checkpoint(path: Path) -> dict[str, object] | RegressionModel:
    """Load a trusted local checkpoint written by train_kappa_10cm.py."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, (dict, RegressionModel)):
        raise ValueError("checkpoint is neither a state dictionary nor a supported RegressionModel")
    return checkpoint


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=directory / "nn_radcal_pl10_v1.pt")
    parser.add_argument("--onnx", type=Path, default=directory / "radcal_pl10_v1.onnx")
    parser.add_argument("--manifest", type=Path, default=directory / "radcal_pl10_v1.json")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        required = {"input_names", "temperature_min", "temperature_max", "species_min", "species_max", "target_name"}
        if required - checkpoint.keys() or tuple(checkpoint["input_names"]) != MODEL_STATE_INPUT_NAMES:
            raise ValueError("new-format checkpoint feature order or metadata does not match the RadCal PL10 contract")
        model = DeploymentModel(checkpoint).eval()
        input_names = MODEL_STATE_INPUT_NAMES
        normalization = "embedded_log1p_minmax"
        valid_range = {
            "temperature_K": [float(checkpoint["temperature_min"]), float(checkpoint["temperature_max"])],
            "species_min": [float(x) for x in checkpoint["species_min"]],
            "species_max": [float(x) for x in checkpoint["species_max"]],
        }
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # This is nn_radcal_pl10_v1.pt. The tensor shapes and module names are
        # intentionally checked before loading, rather than inferred silently.
        state = checkpoint["model_state_dict"]
        required = {"normalization.mean", "normalization.variance", "normalization.std", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        if not isinstance(state, dict) or required - state.keys() or tuple(state["fc1.weight"].shape) != (32, 6):
            raise ValueError("legacy checkpoint does not match the supported 6-to-32-to-1 PL10 architecture")
        model = LegacyPL10Model()
        model.load_state_dict(state)
        model.eval()
        input_names = MODEL_STATE_INPUT_NAMES
        normalization = "embedded_zscore"
        valid_range = {"normalization_mean": [float(x) for x in checkpoint["normalization_mean"]],
                       "normalization_variance": [float(x) for x in checkpoint["normalization_variance"]]}
    elif isinstance(checkpoint, RegressionModel):
        required_modules = {"normalization", "fc1", "fc2"}
        if required_modules - checkpoint._modules.keys() or checkpoint.fc1.in_features != 6 or checkpoint.fc1.out_features != 32 or checkpoint.fc2.in_features != 32 or checkpoint.fc2.out_features != 1:
            raise ValueError("full checkpoint does not match the supported 6-to-32-to-1 RadCal architecture")
        if not all(hasattr(checkpoint.normalization, name) for name in ("mean", "variance", "std")):
            raise ValueError("full checkpoint is missing its z-score normalization parameters")
        model = checkpoint.eval()
        input_names = FULL_MODEL_INPUT_NAMES
        normalization = "embedded_zscore"
        valid_range = {
            "normalization_mean": [float(x) for x in checkpoint.normalization.mean],
            "normalization_variance": [float(x) for x in checkpoint.normalization.variance],
        }
    else:
        raise ValueError("checkpoint has neither 'model_state' nor 'model_state_dict'")
    example = torch.tensor([[1000.0, 0.10, 0.10, 0.01, 0.00, 1.0e-5]], dtype=torch.float32)
    with torch.inference_mode():
        reference = model(example)
        torch.onnx.export(
            model, example, args.onnx, opset_version=args.opset,
            input_names=["radiation_features"], output_names=["kappa_10cm_cm_1"],
            dynamic_axes={"radiation_features": {0: "batch"}, "kappa_10cm_cm_1": {0: "batch"}},
        )
    try:
        import onnx
        onnx.checker.check_model(str(args.onnx))
    except ImportError:
        print("Warning: install 'onnx' to perform structural validation.")

    manifest = {
        "model_id": "radcal-pl10", "model_version": "1.0.0", "format": "onnx",
        "model_file": args.onnx.name, "sha256": checksum(args.onnx),
        "input": {"tensor_name": "radiation_features", "shape": ["batch", 6],
                  "features": list(input_names), "units": ["K", "1", "1", "1", "1", "1"],
                  "normalization": normalization, "normalization_parameters": valid_range},
        "output": {"tensor_name": "kappa_10cm_cm_1", "shape": ["batch", 1],
                   "quantity": "direct_radcal_kappa_10cm", "units": "cm^-1",
                   "fds_units": "m^-1", "fds_scale_factor": 100.0,
                   "supported_radiation_bands": 1},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported: {args.onnx}")
    print(f"Manifest: {args.manifest}")
    print(f"PyTorch example output (cm^-1): {reference.item():.8e}")


if __name__ == "__main__":
    main()
