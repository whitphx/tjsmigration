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
    rename_base_model_to: str | None = None  # If provided, the base model will be renamed to this name


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

        # Transformers.js v3 expects `model.onnx` instead of `decoder_model_merged.onnx` in case of decoder-only models.
        has_encoder = len(list((temp_dir / "onnx").glob("encoder_*.onnx"))) > 0
        if not has_encoder:
            decoder_model_path = temp_dir / "onnx" / "decoder_model_merged.onnx"
            if decoder_model_path.exists():
                new_decoder_model_path = temp_dir / "onnx" / "model.onnx"
                logger.info(f"Renaming {decoder_model_path} to {new_decoder_model_path}")
                decoder_model_path.rename(new_decoder_model_path)

        # 2. Merge the target files into the temporary directory
        onnx_dir = temp_dir / "onnx"
        def add_onnx_file(file_path: Path):
            shutil.copy(file_path, onnx_dir / file_path.name)

        yield temp_dir, add_onnx_file


def run_js_pipeline_e2e_test(
        task_name: str,
        model_dir: Path,  # e.g. /path/to/onnx-community/whisper-tiny
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
    has_encoder = any(basename.startswith("encoder_") for basename in base_model_basenames)
    quantization_configs = []
    for base_model_basename in base_model_basenames:
        config = get_quantization_config_for_base_model(onnx_dir, base_model_basename)
        if not has_encoder and base_model_basename == "decoder_model_merged":
            # Some transformers.js v2 decoder-only models used the older `decoder_model_merged*.onnx` name,
            # but now it's simply `model.onnx` because we don't need to export multiple decoders and merge anymore.
            if "model" in base_model_basenames:
                continue
            else:
                config.rename_base_model_to = "model.onnx"
        quantization_configs.append(config)
    return quantization_configs


@dataclass
class QuantizedModelInfo:
    mode: str
    reason: Literal["missing", "invalid"]
    path: Path | None
    external_path: Path | None
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


def call_quantization_script(
        hf_api: HfApi,
        model_info: ModelInfo,
        quantization_config: QuantizationConfig,
        working_dir: Path,
        output_dir: Path,
        ignore_done_task_type_inference_failure: bool,
) -> QuantizationResult:
    with tempfile.TemporaryDirectory() as temp_input_dir:
        base_model = quantization_config.base_model

        # Copy the base model file into the temp input directory with or without slimming by onnxslim
        # Rename the file as well if `quantization_config.rename_base_model_to` is set.
        temp_source_file_name = quantization_config.rename_base_model_to or base_model.name
        temp_source_file_path = Path(temp_input_dir) / temp_source_file_name
        if quantization_config.slim:
            logger.info(f"Slimming {base_model} to {temp_source_file_path}")
            cmd = [
                "uv", "run",
                "--with-requirements", "scripts/requirements.txt",
                "onnxslim", str(base_model), str(temp_source_file_path)
            ]
            subprocess.run(cmd, cwd=TRANSFORMERS_JS_PATH, check=True)
        else:
            logger.info(f"Copying {base_model} to {temp_source_file_path}")
            shutil.copy(base_model, temp_source_file_path)

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
        if task_name is None and not ignore_done_task_type_inference_failure:
            raise ValueError(f"Task type for {model_info.id} could not be inferred. Please specify the task type manually or check the model repository.")

        # Validate the quantized model
        logger.info("Validating quantized models...")
        with prepare_js_e2e_test_directory(hf_api, model_info.id) as (temp_dir, add_onnx_file):
            results = []
            for quantization in quantization_config.quantizations:
                model_base_name = Path(temp_source_file_name).stem
                suffix = get_quantized_model_suffix(quantization.type)
                quantized_model_file_name = f"{model_base_name}_{suffix}.onnx"
                quantized_model_path = working_dir / quantized_model_file_name
                if not quantized_model_path.exists():
                    raise FileNotFoundError(f"Quantized model {quantized_model_path} not found")
                external_file_path = working_dir / (quantized_model_file_name + "_data") # path/to/model.onnx_data, Ref: https://github.com/huggingface/transformers.js/blob/28852a2ad92e9bf410af11fe03e2b8b51e96c0d6/scripts/utils.py#L53-L54
                if not external_file_path.exists():
                    external_file_path = None

                if validate_onnx_model(quantized_model_path):
                    logger.info(f"{quantized_model_path.name}: ONNX check passed ✳️")

                    logger.info(f"Copying {quantized_model_path} to JS-based E2E test environment")
                    add_onnx_file(quantized_model_path)
                    if external_file_path:
                        logger.info(f"Copying {external_file_path} to JS-based E2E test environment")
                        add_onnx_file(external_file_path)

                    if task_name is None:
                        logger.warning(f"⚠️ Task type for {model_info.id} could not be inferred. Skipping JS-based E2E test.")
                        results.append(QuantizedModelInfo(mode=quantization.type, reason=quantization.reason, path=quantized_model_path, status="success", e2e_test_error_message=None))
                        continue

                    success, error_message = run_js_pipeline_e2e_test(
                        task_name=task_name,
                        model_dir=temp_dir,
                        model_base_name=model_base_name,
                        quantization_type=quantization.type
                    )
                    if success:
                        logger.info(f"{quantized_model_path.name}: JS-based E2E test passed ✳️")
                        output_quantized_model_path = output_dir / quantized_model_path.name
                        logger.info(f"Copying {quantized_model_path} to {output_quantized_model_path}")
                        shutil.copy(quantized_model_path, output_quantized_model_path)
                        if external_file_path:
                            output_external_model_path = output_dir / external_file_path.name
                            logger.info(f"Copying {external_file_path} to {output_external_model_path}")
                            shutil.copy(external_file_path, output_external_model_path)
                        else:
                            output_external_model_path = None
                        results.append(QuantizedModelInfo(mode=quantization.type, reason=quantization.reason, path=output_quantized_model_path, external_path=output_external_model_path, status="success", e2e_test_error_message=None))
                    else:
                        logger.warning(f"{quantized_model_path.name}: JS-based E2E test failed ❌")
                        results.append(QuantizedModelInfo(mode=quantization.type, reason=quantization.reason, path=None, external_path=None, status="js_e2e_test_failed", e2e_test_error_message=error_message))
                else:
                    logger.warning(f"{quantized_model_path.name}: ONNX check failed ❌")
                    results.append(QuantizedModelInfo(mode=quantization.type, reason=quantization.reason, path=None, external_path=None, status="onnx_check_failed", e2e_test_error_message=None))

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
        if result.config.rename_base_model_to:
            summary += f"**The base model `{result.config.base_model.name}` has been renamed to `{result.config.rename_base_model_to}`.**\n\n"
        if not success:
            summary += f"```\n{result.error}\n```\n"
        for model_info in result.models:
            summary += f"↳ {'✅' if model_info.status == "success" else '❌'} `{model_info.mode}`: `{model_info.path.name}` ({create_reason_text(model_info.reason)}{" but " + create_failed_status_text(model_info.status) if model_info.status != "success" else ""})\n"
            if model_info.status == "js_e2e_test_failed":
                summary += f"```\n{model_info.e2e_test_error_message}\n```\n"
        summary += "\n"
    return summary


def migrate_model_files(
        hf_api: HfApi,
        model_info: ModelInfo,
        working_dir: Path,
        output_dir: Path,
        fallback_to_no_slimming: bool = True,
        ignore_done_task_type_inference_failure: bool = False,
) -> str:
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
        result = call_quantization_script(
            hf_api=hf_api,
            model_info=model_info,
            quantization_config=quantization_config,
            working_dir=working_dir,
            output_dir=output_dir,
            ignore_done_task_type_inference_failure=ignore_done_task_type_inference_failure
        )
        results.append(result)
        if result.error and fallback_to_no_slimming:
            logger.warning(f"Failed to quantize {quantization_config.base_model.stem} with slimming. Retrying without slimming...")
            quantization_config = dataclasses.replace(quantization_config, slim=False)
            result = call_quantization_script(
                hf_api=hf_api,
                model_info=model_info,
                quantization_config=quantization_config,
                working_dir=working_dir,
                output_dir=output_dir,
                ignore_done_task_type_inference_failure=ignore_done_task_type_inference_failure
            )
            results.append(result)

    summary = create_summary_text(results)

    return summary
