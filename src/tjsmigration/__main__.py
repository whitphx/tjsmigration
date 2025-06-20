import os
import logging
import shutil
from pathlib import Path

import click
from huggingface_hub import HfApi
from anthropic import Anthropic

from .model_migration import migrate_model_files
from .readme_migration import migrate_readme
from .tempdir import temp_dir_if_none

logger = logging.getLogger(__name__)


@click.command()
@click.option("--repo-id", required=True)
@click.option("--output-dir", required=False, type=click.Path(exists=False))
@click.option("--upload", required=False, is_flag=True)
@click.option("--only", required=False, multiple=True, type=click.Choice(["readme", "model"]), default=["readme", "model"])
def migrate(repo_id: str, output_dir: str | None, upload: bool, only: list[str]):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set")

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    hf_api = HfApi(token=token)

    repo_info = hf_api.repo_info(repo_id)

    logger.info(f"Migrating {repo_id}...")
    logger.info(f"Upload: {upload}")
    logger.info(f"Only: {only}")

    output_dir = Path(output_dir) if output_dir else None

    anthropic_client = Anthropic(api_key=anthropic_api_key)

    with temp_dir_if_none(output_dir) as output_dir:
        repo_output_dir: Path = output_dir / repo_info.id
        if repo_output_dir.exists():
            logger.warning(f"Output directory {repo_output_dir} already exists. This script will overwrite it.")
            shutil.rmtree(repo_output_dir)
        repo_output_dir.mkdir(parents=True, exist_ok=True)

        repo_onnx_output_dir = repo_output_dir / "onnx"

        if "readme" in only:
            migrate_readme(hf_api=hf_api, anthropic_client=anthropic_client, model_info=repo_info, output_dir=repo_output_dir)
        if "model" in only:
            summary = migrate_model_files(hf_api=hf_api, model_info=repo_info, output_dir=repo_onnx_output_dir)

        logger.info(summary)

        files = [p for p in repo_output_dir.glob("**/*") if p.is_file()]
        logger.info(f"Generated files:\n{'\n'.join([' - ' + str(p) for p in files])}")

        if len(files) == 0:
            logger.warning("No files were created")
            return
        if not upload:
            logger.info("Skipping upload")
            return

        logger.info(f"Uploading quantized models to {repo_id}...")
        commit_info = hf_api.upload_folder(
            repo_id=repo_id,
            folder_path=repo_output_dir,
            repo_type="model",
            commit_message="Add/update the quantized ONNX model files and README.md for Transformers.js v3",
            commit_description=summary,
            create_pr=True,
        )
        logger.info(f"Uploaded quantized models to the Hugging Face Hub: {commit_info}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    migrate()
