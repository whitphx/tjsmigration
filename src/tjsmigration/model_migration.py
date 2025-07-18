import logging
import contextlib
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Literal
import dataclasses
from dataclasses import dataclass

import onnx
from huggingface_hub import HfApi, ModelInfo

from .task_type import infer_transformers_task_type


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ROOT = HERE.parent


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
    slim: bool
    quantizations: list[RequiredQuantization]


def get_base_model_basenames(onnx_dir: Path) -> list[str]:
    base_models = []
    for onnx_file in onnx_dir.glob("*.onnx"):
        basename = onnx_file.stem
        is_quantized = (
            any(basename.endswith(f"_{get_quantized_model_suffix(quantization_type)}") for quantization_type in QUANTIZATION_TYPES)
        )
        if not is_quantized:
            base_models.append(onnx_file.stem)
    return base_models


def parse_quantized_model_filename(filepath: Path) -> tuple[str, str]:
    for quantization_type in QUANTIZATION_TYPES:
        suffix = "_" + get_quantized_model_suffix(quantization_type)
        if filepath.stem.endswith(suffix):
            return filepath.stem.replace(suffix, ""), quantization_type
    return filepath.stem, None


def validate_onnx_model(abspath: Path) -> bool:
    try:
        onnx.checker.check_model(str(abspath), full_check=True)
        return True
    except onnx.onnx_cpp2py_export.checker.ValidationError:
        return False


@contextlib.contextmanager
def prepare_js_e2e_test_directory(
    hf_api: HfApi,
    original_repo_id: str,  # e.g. "onnx-community/whisper-tiny". Metadata files are needed for the model files to be loaded, so we need to copy them from the original repo.
):
    source_repo_path = hf_api.snapshot_download(repo_id=original_repo_id, repo_type="model")

    # 1. Copy the original repo to a temporary directory
    with tempfile.TemporaryDirectory() as root_temp_dir:
        temp_dir = Path(root_temp_dir) / Path(source_repo_path).name
        shutil.copytree(source_repo_path, temp_dir)

        # 2. Merge the target files into the temporary directory
        onnx_dir = temp_dir / "onnx"
        def add_onnx_file(file_path: Path):
            shutil.copy(file_path, onnx_dir / file_path.name)

        yield temp_dir, add_onnx_file


def run_js_e2e_test(
        task_name: str,
        model_dir: Path,  # e.g. /path/to/onnx-communty/whisper-tiny
        model_base_name: str,  # e.g. decoder_model_merged
        quantization_type: str  # e.g. fp16
) -> tuple[bool, str | None]:
    js_code = f"""
import {{ pipeline, env }} from '@huggingface/transformers';
import path from 'node:path';

env.allowLocalModels = true;
env.allowRemoteModels = false;

const model = await pipeline('{task_name}', '{model_dir.resolve()}', {{
    dtype: {{
        '{model_base_name}': '{quantization_type}'
    }}
}});
"""

    try:
        subprocess.run(["node", "-e", js_code], cwd=ROOT, check=True, capture_output=True, text=True)
        return True, None
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip() if e.stderr else str(e)
        return False, error_message


def get_quantization_config_for_base_model(onnx_dir: Path, base_model_basename: str) -> QuantizationConfig:
    base_model_path = onnx_dir / f"{base_model_basename}.onnx"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model {base_model_basename} not found")

    quantizations = []
    for quantization_type in QUANTIZATION_TYPES:
        suffix = get_quantized_model_suffix(quantization_type)
        quantized_model_path = onnx_dir / f"{base_model_basename}_{suffix}.onnx"
        if quantized_model_path.exists():
            if not validate_onnx_model(quantized_model_path):
                quantizations.append(RequiredQuantization(type=quantization_type, reason="invalid"))
        else:
            quantizations.append(RequiredQuantization(type=quantization_type, reason="missing"))
    return QuantizationConfig(base_model=base_model_path, slim=True, quantizations=quantizations)


def get_quantization_configs(onnx_dir: Path) -> list[QuantizationConfig]:
    base_model_basenames = get_base_model_basenames(onnx_dir)
    quantization_configs = []
    for base_model_basename in base_model_basenames:
        quantization_configs.append(get_quantization_config_for_base_model(onnx_dir, base_model_basename))
    return quantization_configs


@dataclass
class QuantizedModelInfo:
    mode: str
    reason: Literal["missing", "invalid"]
    path: Path
    status: Literal["success", "onnx_check_failed", "js_e2e_test_failed"]
    e2e_test_error_message: str | None


@dataclass
class QuantizationResult:
    config: QuantizationConfig
    error: str | None
    models: list[QuantizedModelInfo]

    def success(self) -> bool:
        if self.error:
            return False
        return all(result.status == "success" for result in self.models)


def call_quantization_script(hf_api: HfApi, model_info: ModelInfo, quantization_config: QuantizationConfig, working_dir: Path, output_dir: Path) -> QuantizationResult:
    with tempfile.TemporaryDirectory() as temp_input_dir:
        # Copy the base model file into the temp input directory with or without slimming by onnxslim
        base_model = quantization_config.base_model
        temp_source_file = Path(temp_input_dir) / base_model.name
        if quantization_config.slim:
            logger.info(f"Slimming {base_model} to {temp_source_file}")
            cmd = [
                "uv", "run",
                "--with-requirements", "scripts/requirements.txt",
                "onnxslim", str(base_model), str(temp_source_file)
            ]
            subprocess.run(cmd, cwd=TRANSFORMERS_JS_PATH, check=True)
        else:
            logger.info(f"Copying {base_model} to {temp_source_file}")
            shutil.copy(base_model, temp_source_file)

        # Run the quantization script
        modes = [quantization.type for quantization in quantization_config.quantizations]
        cmd = [
            "uv", "run",
            "--with-requirements", "scripts/requirements.txt",
            "python", "-m", "scripts.quantize",
            "--input_folder", temp_input_dir,
            "--output_folder", str(working_dir.resolve()),
            "--modes", *modes,
        ]
        logger.info(f"Running command: {cmd}")
        try:
            subprocess.run(cmd, cwd=TRANSFORMERS_JS_PATH, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip() if e.stderr else str(e)
            logger.error(f"Error running quantization script: {error_message}")
            return QuantizationResult(config=quantization_config, models=[], error=error_message)

        task_name = infer_transformers_task_type(model_info)
        # Validate the quantized model
        logger.info("Validating quantized models...")
        with prepare_js_e2e_test_directory(hf_api, model_info.id) as (temp_dir, add_onnx_file):
            results = []
            for quantization in quantization_config.quantizations:
                suffix = get_quantized_model_suffix(quantization.type)
                quantized_model_path = working_dir / f"{base_model.stem}_{suffix}.onnx"
                if not quantized_model_path.exists():
                    raise FileNotFoundError(f"Quantized model {quantized_model_path} not found")

                if validate_onnx_model(quantized_model_path):
                    logger.info(f"{quantized_model_path.name}: ONNX check passed ✳️")
                    add_onnx_file(quantized_model_path)
                    success, error_message = run_js_e2e_test(task_name, temp_dir, base_model.stem, quantization.type)
                    if success:
                        logger.info(f"{quantized_model_path.name}: JS-based E2E test passed ✳️")
                        logger.info(f"Copying {quantized_model_path} to {output_dir / quantized_model_path.name}")
                        output_path = output_dir / quantized_model_path.name
                        shutil.copy(quantized_model_path, output_path)
                        results.append(QuantizedModelInfo(mode=quantization.type, reason=quantization.reason, path=output_path, status="success", e2e_test_error_message=None))
                    else:
                        logger.warning(f"{quantized_model_path.name}: JS-based E2E test failed ❌")
                        results.append(QuantizedModelInfo(mode=quantization.type, reason=quantization.reason, path=quantized_model_path, status="js_e2e_test_failed", e2e_test_error_message=error_message))
                else:
                    logger.warning(f"{quantized_model_path.name}: ONNX check failed ❌")
                    results.append(QuantizedModelInfo(mode=quantization.type, reason=quantization.reason, path=quantized_model_path, status="onnx_check_failed", e2e_test_error_message=None))

        return QuantizationResult(config=quantization_config, models=results, error=None)


def create_reason_text(reason: Literal["missing", "invalid"]) -> str:
    if reason == "missing":
        return "added"
    elif reason == "invalid":
        return "replaced because it was invalid"
    else:
        return ""


def create_failed_status_text(status: Literal["onnx_check_failed", "js_e2e_test_failed"]) -> str:
    if status == "onnx_check_failed":
        return "ONNX check failed"
    elif status == "js_e2e_test_failed":
        return "JS-based E2E test failed"
    else:
        return ""


def create_summary_text(results: list[QuantizationResult]) -> str:
    summary = "## Applied Quantizations\n\n"

    if len(results) == 0:
        summary += "**No new quantized models were added.**\n\n"
        return summary

    for result in results:
        success = result.success()
        summary += f"### {'✅' if success else '❌'} Based on `{result.config.base_model.name}` *{'with' if result.config.slim else 'without'}* slimming\n\n"
        if not success:
            summary += f"```\n{result.error}\n```\n"
        for model_info in result.models:
            summary += f"↳ {'✅' if model_info.status == "success" else '❌'} `{model_info.mode}`: `{model_info.path.name}` ({create_reason_text(model_info.reason)}{" but " + create_failed_status_text(model_info.status) if model_info.status != "success" else ""})\n"
            if model_info.status == "js_e2e_test_failed":
                summary += f"```\n{model_info.e2e_test_error_message}\n```\n"
        summary += "\n"
    return summary


def migrate_model_files(hf_api: HfApi, model_info: ModelInfo, working_dir: Path, output_dir: Path, fallback_to_no_slimming: bool = True) -> str:
    repo_id = model_info.id

    downloaded_path = hf_api.snapshot_download(repo_id=repo_id, repo_type="model")

    src_onnx_dir = Path(downloaded_path) / "onnx"
    quantization_configs = get_quantization_configs(src_onnx_dir)

    if len(list(output_dir.glob("*.onnx"))) > 0:
        raise ValueError("Output directory already contains some files. Abort.")
    if len(list(working_dir.glob("*.onnx"))) > 0:
        raise ValueError("Working directory already contains some files. Abort.")

    logger.info("Quantizing models...")
    logger.info(f"Quantization configs: {quantization_configs}")
    results = []
    for quantization_config in quantization_configs:
        if len(quantization_config.quantizations) == 0:
            logger.warning(f"No new quantization configs needed for {quantization_config.base_model.stem}. Skipping...")
            continue
        quantization_config.slim = True
        result = call_quantization_script(hf_api=hf_api, model_info=model_info, quantization_config=quantization_config, working_dir=working_dir, output_dir=output_dir)
        results.append(result)
        if result.error and fallback_to_no_slimming:
            logger.warning(f"Failed to quantize {quantization_config.base_model.stem} with slimming. Retrying without slimming...")
            quantization_config = dataclasses.replace(quantization_config, slim=False)
            result = call_quantization_script(hf_api=hf_api, model_info=model_info, quantization_config=quantization_config, working_dir=working_dir, output_dir=output_dir)
        results.append(result)

    summary = create_summary_text(results)

    return summary
