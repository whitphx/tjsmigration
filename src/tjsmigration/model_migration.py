import os
import contextlib
import logging
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Generator, Literal
from dataclasses import dataclass

import onnx
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


QUANTIZATION_TYPES = [
    "fp16",
    "q8",
    "int8",
    "uint8",
    "q4",
    "q4f16",
    "bnb4",
]

QUANTIZE_SUFFIX_MAPPING = {
    "q8": "quantized",
}

def get_quantized_model_suffix(quantization_type: str) -> str:
    return QUANTIZE_SUFFIX_MAPPING.get(quantization_type, quantization_type)


TRANSFORMERS_JS_PATH = "./transformers.js"


@dataclass
class RequiredQuantization:
    type: str
    reason: Literal["missing", "invalid"]


@dataclass
class QuantizationConfig:
    base_model: Path
    quantizations: list[RequiredQuantization]


def get_base_model_basenames(onnx_dir: Path) -> list[str]:
    base_models = []
    for onnx_file in onnx_dir.glob("*.onnx"):
        basename = onnx_file.stem
        is_quantized = (
            any(basename.endswith(f"_{quantization_type}") for quantization_type in QUANTIZATION_TYPES)
            or basename.endswith("_quantized")
        )
        if not is_quantized:
            base_models.append(onnx_file.stem)
    return base_models


def get_quantization_config_for_base_model(onnx_dir: Path, base_model_basename: str) -> QuantizationConfig:
    base_model_path = onnx_dir / f"{base_model_basename}.onnx"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model {base_model_basename} not found")

    quantizations = []
    for quantization_type in QUANTIZATION_TYPES:
        suffix = get_quantized_model_suffix(quantization_type)
        quantized_model_path = onnx_dir / f"{base_model_basename}_{suffix}.onnx"
        if quantized_model_path.exists():
            try:
                onnx.checker.check_model(str(quantized_model_path), full_check=True)
            except onnx.onnx_cpp2py_export.checker.ValidationError:
                quantizations.append(RequiredQuantization(type=quantization_type, reason="invalid"))
        else:
            quantizations.append(RequiredQuantization(type=quantization_type, reason="missing"))
    return QuantizationConfig(base_model=base_model_path, quantizations=quantizations)


def get_quantization_configs(onnx_dir: Path) -> list[QuantizationConfig]:
    base_model_basenames = get_base_model_basenames(onnx_dir)
    quantization_configs = []
    for base_model_basename in base_model_basenames:
        quantization_configs.append(get_quantization_config_for_base_model(onnx_dir, base_model_basename))
    return quantization_configs


@contextlib.contextmanager
def temp_dir_if_none(dir_path: Path | None) -> Generator[Path, None, None]:
    if dir_path is None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    else:
        yield dir_path


@dataclass
class QuantizationResult:
    success: bool
    config: QuantizationConfig


def call_quantization_script(quantization_config: QuantizationConfig, output_dir: Path) -> QuantizationResult:
    with tempfile.TemporaryDirectory() as temp_input_dir:
        # copy the base model file into the temp input directory
        base_model = quantization_config.base_model
        shutil.copy(base_model, temp_input_dir)

        modes = [quantization.type for quantization in quantization_config.quantizations]

        cmd = [
            "uv", "run",
            "--with-requirements", "scripts/requirements.txt",
            "python", "-m", "scripts.quantize",
            "--input_folder", temp_input_dir,
            "--output_folder", str(output_dir.resolve()),
            "--modes", *modes,
        ]

        logger.info(f"Running command: {cmd}")
        try:
            subprocess.run(cmd, cwd=TRANSFORMERS_JS_PATH)
            return QuantizationResult(success=True, config=quantization_config)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running quantization script: {e}")
            return QuantizationResult(success=False, config=quantization_config)



def create_reason_text(reason: Literal["missing", "invalid"]) -> str:
    if reason == "missing":
        return "added"
    elif reason == "invalid":
        return "replaced because it was invalid"
    else:
        return ""


def create_summary_text(results: list[QuantizationResult]) -> str:
    summary = "### Applied Quantizations\n"
    for result in results:
        summary += f"#### {'✅' if result.success else '❌'} `{result.config.base_model.stem}.onnx`\n"
        for quantization in result.config.quantizations:
            summary += f"- `{quantization.type}` ({create_reason_text(quantization.reason)})\n"
        summary += "\n"
    return summary


def migrate_model_files(hf_api: HfApi, repo_id: str, output_dir: Path, upload: bool):
    downloaded_path = hf_api.snapshot_download(repo_id=repo_id, repo_type="model")

    onnx_dir = Path(downloaded_path) / "onnx"
    quantization_configs = get_quantization_configs(onnx_dir)

    results = []
    with temp_dir_if_none(output_dir) as output_dir:
        file_exists = len(list(output_dir.glob("*.onnx"))) > 0
        if file_exists:
            raise ValueError("Output directory already contains some files. Abort.")

        logger.info("Quantizing models...")
        logger.info(f"Quantization configs: {quantization_configs}")
        for quantization_config in quantization_configs:
            result = call_quantization_script(quantization_config, output_dir)
            results.append(result)

        summary = create_summary_text(results)
        logger.info(summary)

        new_file_exists = len(list(output_dir.glob("*.onnx"))) > 0

        if upload:
            if new_file_exists:
                logger.info(f"Uploading quantized models to {repo_id}...")
                commit_info = hf_api.upload_folder(
                    repo_id=repo_id,
                    folder_path=output_dir,
                    path_in_repo="onnx",
                    repo_type="model",
                    commit_message="Add quantized ONNX model files",
                    create_pr=True,
                )
                logger.info(f"Uploaded quantized models to the Hugging Face Hub: {commit_info}")
            else:
                logger.warning("No quantized models were created")
        else:
            logger.info("Skipping upload")
