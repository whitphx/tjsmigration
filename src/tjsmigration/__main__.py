import os
from datetime import datetime
import logging
import shutil
import json
from pathlib import Path

import click
from huggingface_hub import HfApi
from anthropic import Anthropic

from .model_migration import migrate_model_files, parse_quantized_model_filename, prepare_js_e2e_test_directory, validate_onnx_model, run_js_e2e_test
from .readme_migration import migrate_readme
from .tempdir import temp_dir_if_none
from .task_type import infer_transformers_task_type

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ROOT = HERE.parent.parent


@click.group()
def cli():
    pass


def get_user_confirmation_to_upload(repo_id: str, files: list[Path], summary: str) -> bool:
    text = ""
    text += f"=== Upload Confirmation ===\n"
    text += f"Repo ID: {repo_id}\n"
    text += f"Generated files:\n{'\n'.join([' - ' + str(p) for p in files])}\n"
    text += "\n-- Summary --\n"
    text += summary + "\n"
    text += "\n-- End of Summary --\n"
    text += "\n\nDo you want to upload the files to the Hugging Face Hub?"
    return click.confirm(text)


def migrate_repo(hf_api: HfApi, anthropic_client: Anthropic, repo_id: str, output_dir_path: str | None, working_dir_path: str | None, upload: bool, only: list[str], result_file_path: Path | None):
    logger.info(f"Migrating {repo_id}...")
    logger.info(f"Upload: {upload}")
    logger.info(f"Only: {only}")

    repo_info = hf_api.repo_info(repo_id)

    output_dir = Path(output_dir_path) if output_dir_path else None
    working_dir = Path(working_dir_path) if working_dir_path else None

    with temp_dir_if_none(output_dir) as output_dir, temp_dir_if_none(working_dir) as working_dir:
        repo_output_dir: Path = output_dir / repo_info.id
        if repo_output_dir.exists():
            logger.warning(f"Output directory {repo_output_dir} already exists. This script will overwrite it.")
            shutil.rmtree(repo_output_dir)
        repo_output_dir.mkdir(parents=True, exist_ok=True)

        repo_working_dir = working_dir / repo_info.id
        if repo_working_dir.exists():
            logger.warning(f"Working directory {repo_working_dir} already exists. This script will overwrite it.")
            shutil.rmtree(repo_working_dir)
        repo_working_dir.mkdir(parents=True, exist_ok=True)

        repo_onnx_output_dir = repo_output_dir / "onnx"
        repo_onnx_working_dir = repo_working_dir / "onnx"

        if "readme" in only:
            migrate_readme(hf_api=hf_api, anthropic_client=anthropic_client, model_info=repo_info, output_dir=repo_output_dir)
        if "model" in only:
            repo_onnx_working_dir.mkdir(parents=True, exist_ok=True)
            repo_onnx_output_dir.mkdir(parents=True, exist_ok=True)
            summary = migrate_model_files(hf_api=hf_api, model_info=repo_info, working_dir=repo_onnx_working_dir, output_dir=repo_onnx_output_dir)

        logger.info(summary)

        files = [p for p in repo_output_dir.glob("**/*") if p.is_file()]
        logger.info(f"Generated files:\n{'\n'.join([' - ' + str(p) for p in files])}")

        if len(files) == 0:
            logger.warning("No files were created")
            return
        if not upload:
            logger.info("Skipping upload")
            return

        if not get_user_confirmation_to_upload(repo_id, files, summary):
            logger.info("Upload cancelled by user")
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
        logger.info(f"Uploaded files to the Hugging Face Hub")

        print(f"Pull request created: {commit_info.pr_url}")

        if result_file_path:
            with result_file_path.open("w") as f:
                json.dump(
                    {
                        "repo_id": repo_id,
                        "pr_url": commit_info.pr_url,
                    },
                    f,
                )


@cli.command()
@click.option("--repo", required=False, multiple=True, type=str)
@click.option("--author", required=False, type=str)
@click.option("--model-name", required=False, type=str)
@click.option("--filter", required=False, multiple=True, type=str)
@click.option("--output-dir", required=False, type=click.Path(exists=False))
@click.option("--working-dir", required=False, type=click.Path(exists=False))
@click.option("--upload", required=False, is_flag=True)
@click.option("--only", required=False, multiple=True, type=click.Choice(["readme", "model"]), default=["readme", "model"])
def migrate(repo: tuple[str], author: str | None, model_name: str | None, filter: tuple[str], output_dir: str | None, working_dir: str | None, upload: bool, only: list[str]):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set")

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    hf_api = HfApi(token=token)
    anthropic_client = Anthropic(api_key=anthropic_api_key)

    repo = list(repo)

    if author or model_name or filter:
        search_results = hf_api.list_models(library="transformers.js", author=author, model_name=model_name, filter=filter)
        searched_repo_ids = [r.id for r in search_results]
        repo = repo + searched_repo_ids

    logger.info(f"Target repos:\n{'\n'.join([' - ' + r for r in repo])}")
    if not click.confirm("Are you sure you want to migrate these repos?"):
        logger.info("Migration cancelled by user")
        return

    logger.info(f"Migrating {repo}...")

    result_file_path = ROOT / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for repo_id in repo:
        migrate_repo(hf_api=hf_api, anthropic_client=anthropic_client, repo_id=repo_id, output_dir_path=output_dir, working_dir_path=working_dir, upload=upload, only=only, result_file_path=result_file_path)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def check(path: str):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path {path} does not exist")
    if not path.is_file():
        raise ValueError(f"Path {path} is not a file")

    if not validate_onnx_model(path):
        raise ValueError(f"Path {path} is not a valid ONNX model")


@cli.command()
@click.option("--repo", required=True, help="The Hugging Face repo ID from which the model files are downloaded")
@click.argument("path", type=click.Path(exists=True), nargs=-1)
def test(repo: str, path: list[str]):
    paths = [Path(p) for p in path]
    for p in paths:
        if not p.exists():
            raise ValueError(f"Path {p} does not exist")
        if not p.is_file():
            raise ValueError(f"Path {p} is not a file")
        if not p.suffix == ".onnx":
            raise ValueError(f"Path {p} is not an ONNX model")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set")
    hf_api = HfApi(token=token)

    model_info = hf_api.repo_info(repo)
    task_name = infer_transformers_task_type(model_info)

    with prepare_js_e2e_test_directory(hf_api, repo) as (temp_dir, add_onnx_file):
        for p in paths:
            add_onnx_file(p)

            base_model_name, quantization_type = parse_quantized_model_filename(p)

            logger.info(f"Running JS E2E test for {p}, quantized version of [{base_model_name}] with [{quantization_type}] mode...")
            success, error_message = run_js_e2e_test(task_name, temp_dir, base_model_name, quantization_type)
            if not success:
                print(f"Failed to run JS E2E test for {p}: {error_message}")
            else:
                print(f"Successfully ran JS E2E test for {p}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cli()
