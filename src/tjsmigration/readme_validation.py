from pathlib import Path
import logging
import subprocess

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from huggingface_hub import HfApi, ModelInfo

from .task_type import infer_transformers_task_type
from .model_migration import prepare_js_e2e_test_directory


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ROOT = HERE.parent


class ExtractedCodeBlocks(BaseModel):
    """Collection of extracted code blocks with metadata"""
    blocks: list[str] = Field(description="List of extracted code blocks")


def extract_sample_code_blocks(content: str) -> list[str]:
    agent = Agent(
        model="claude-3-5-haiku-latest",
        output_type=ExtractedCodeBlocks,
        system_prompt="""
You are an expert code analyzer. Your task is to extract JavaScript code blocks from README.md content that contain working Transformers.js examples.

## What to Extract
Extract ONLY code blocks that meet ALL of these criteria:
1. Wrapped in triple backticks (```) with optional language specifier like `javascript`, `js`, or no specifier
   - DO NOT include the backticks in the output.
2. Contains a complete, runnable Transformers.js example with:
   - Import statement from '@huggingface/transformers' (e.g., `import { pipeline } from '@huggingface/transformers'`)
   - Pipeline creation and execution
   - Complete example that can be run end-to-end

## What NOT to Extract
- DO NOT extract Installation instructions (npm install, yarn add, etc.)
- DO NOT extract Configuration snippets without pipeline execution
- DO NOT extract Code blocks that only show imports without running the pipeline
- DO NOT extract Non-JavaScript code blocks
- DO NOT extract Incomplete examples that don't execute
- DO NOT extract Code snippets that are fragments of larger examples (e.g., loop iterations like `for (let result of results)` without context)
- DO NOT extract Code blocks that lack essential setup like imports or pipeline initialization
- DO NOT extract Partial examples that assume code was executed before the shown snippet
""",
    )
    result = agent.run_sync(content)
    return result.output.blocks


def run_js_e2e_test_on_sample_code(
    sample_code: str,
    task_type: str | None
):
    setup_code = """
import { env } from '@huggingface/transformers';

env.allowLocalModels = true;
"""
    if task_type != "text-to-speech" and task_type is not None:
        # text-to-speech pipeline imports a remote model (https://github.com/huggingface/transformers.js/blob/8d6c400438df42e1828908e06fa03342c4465129/src/pipelines.js#L2890)
        setup_code += """
env.allowRemoteModels = false;

"""

    js_code = setup_code + sample_code

    if task_type in ["audio-classification", "automatic-speech-recognition", "text-to-audio", "text-to-speech"]:
        expected_error = "Unable to load audio from path/URL since `AudioContext` is not available in your environment."  # It doesn't work in Node.js environment
    else:
        expected_error = None

    logger.debug("Running JS E2E test with code:\n%s", js_code)

    try:
        subprocess.run(["node", "-e", js_code], cwd=ROOT, check=True, capture_output=True, text=True)
        return True, None
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip() if e.stderr else str(e)
        if expected_error and expected_error in error_message:
            return True, error_message
        return False, error_message


def e2e_readme_samples(
    hf_api: HfApi,
    model_info: ModelInfo,
    readme_content: str,
    model_override_dir: Path | None = None,
):
    repo_id = model_info.id
    task_type = infer_transformers_task_type(model_info=model_info)

    sample_code_blocks = extract_sample_code_blocks(readme_content)

    with prepare_js_e2e_test_directory(hf_api, repo_id) as (temp_dir, add_onnx_file):
        if model_override_dir:
            for p in model_override_dir.glob("**/*.onnx"):
                logger.info(f"Adding ONNX file to the E2E test directory: {p}")
                add_onnx_file(p)

        for sample_code_block in sample_code_blocks:
            logger.info(f"Checking sample code block:\n{"\n".join([f"  {line}" for line in sample_code_block.split('\n')])}")
            if not model_info.id in sample_code_block:
                raise ValueError(f"Sample code block does not contain the model ID: {model_info.id}. Something went wrong.")
            sample_code_block = sample_code_block.replace(model_info.id, str(temp_dir))
            success, error_message = run_js_e2e_test_on_sample_code(sample_code_block, task_type)
            if not success:
                raise ValueError(f"Failed to run JS E2E test for sample code block: {error_message}")
