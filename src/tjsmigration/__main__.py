import os
import re
from dataclasses import dataclass
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
from .readme_validation import e2e_readme_samples
from .tempdir import temp_dir_if_none
from .task_type import infer_transformers_task_type

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
ROOT = HERE.parent.parent


LOG_FILE_PATH = ROOT / f"log.json"
FAILED_LOG_FILE_PATH = ROOT / f"failed.json"


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


def migrate_repo(hf_api: HfApi, anthropic_client: Anthropic, repo_id: str, output_dir_path: str | None, working_dir_path: str | None, upload: bool, only: list[str], log_file_path: Path | None, auto: bool):
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
            migrate_readme(
                hf_api=hf_api,
                anthropic_client=anthropic_client,
                model_info=repo_info,
                output_dir=repo_output_dir,
                auto=auto,
            )
        if "model" in only:
            repo_onnx_working_dir.mkdir(parents=True, exist_ok=True)
            repo_onnx_output_dir.mkdir(parents=True, exist_ok=True)
            summary = migrate_model_files(
                hf_api=hf_api,
                model_info=repo_info,
                working_dir=repo_onnx_working_dir,
                output_dir=repo_onnx_output_dir
            )
            logger.info("Model migration summary:\n" + summary)

        files = [p for p in repo_output_dir.glob("**/*") if p.is_file()]
        logger.info(f"Generated files:\n{'\n'.join([' - ' + str(p) for p in files])}")

        if len(files) == 0:
            logger.warning("No files were created")
            return

        if "readme" in only:
            logger.info(f"Run E2E test for sample code blocks in README.md...")
            with open(repo_output_dir / "README.md", "r") as f:
                readme_content = f.read()
            e2e_readme_samples(
                hf_api=hf_api,
                model_info=repo_info,
                model_override_dir=repo_output_dir,
                readme_content=readme_content,
            )

        if not upload:
            logger.info("Skipping upload")
            return

        if not auto:
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

        if log_file_path:
            with log_file_path.open("a") as f:
                json.dump(
                    {
                        "repo_id": repo_id,
                        "pr_url": commit_info.pr_url,
                        "datetime": datetime.now().isoformat(),
                    },
                    f,
                )
                f.write("\n")


@cli.command()
@click.option("--repo", required=False, multiple=True, type=str)
@click.option("--author", required=False, type=str)
@click.option("--model-name", required=False, type=str)
@click.option("--task", required=False, type=str)
@click.option("--filter", required=False, multiple=True, type=str)
@click.option("--exclude", required=False, multiple=True, type=str)
@click.option("--output-dir", required=False, type=click.Path(exists=False))
@click.option("--working-dir", required=False, type=click.Path(exists=False))
@click.option("--upload", required=False, is_flag=True)
@click.option("--only", required=False, multiple=True, type=click.Choice(["readme", "model"]), default=["readme", "model"])
@click.option("--auto", required=False, is_flag=True)
@click.option("--ignore-done", required=False, is_flag=True)
def migrate(
    repo: tuple[str],
    author: str | None,
    model_name: str | None,
    task: str | None,
    filter: tuple[str],
    exclude: tuple[str],
    output_dir: str | None,
    working_dir: str | None,
    upload: bool,
    only: list[str],
    auto: bool,
    ignore_done: bool,
):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set")

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    if auto and not upload:
        if not click.confirm("Are you sure you want to run in auto mode without uploading?"):
            logger.info("Migration cancelled by user")
            return

    hf_api = HfApi(token=token)
    anthropic_client = Anthropic(api_key=anthropic_api_key)

    repo: list[str] = list(repo)

    if author or model_name or task or filter:
        search_results = hf_api.list_models(
            library="transformers.js",
            author=author,
            model_name=model_name,
            pipeline_tag=task,
            filter=filter,
        )
        searched_repo_ids = [r.id for r in search_results]
        repo = repo + searched_repo_ids

    if exclude:
        repo = [r for r in repo if r not in exclude]

    if not ignore_done:
        done_repo_ids = []
        if LOG_FILE_PATH.exists():
            logger.info(f"Loading done repo IDs from {LOG_FILE_PATH}...")
            with LOG_FILE_PATH.open("r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    current_log = json.loads(line)
                    done_repo_ids.append(current_log["repo_id"])
            logger.info(f"Loaded {len(done_repo_ids)} done repo IDs from {LOG_FILE_PATH}")

        repo = [r for r in repo if r not in done_repo_ids]

    if len(repo) == 0:
        logger.info("No repos to migrate")
        return

    logger.info(f"Target repos ({len(repo)}):\n{'\n'.join([' - ' + r for r in repo])}")
    if not auto:
        if not click.confirm("Are you sure you want to migrate these repos?"):
            logger.info("Migration cancelled by user")
            return

    logger.info(f"Migrating {repo}...")

    for repo_id in repo:
        try:
            migrate_repo(hf_api=hf_api, anthropic_client=anthropic_client, repo_id=repo_id, output_dir_path=output_dir, working_dir_path=working_dir, upload=upload, only=only, log_file_path=LOG_FILE_PATH, auto=auto)
        except Exception as e:
            logger.error(f"Error migrating {repo_id}: {e}", exc_info=True)
            if not auto:
                if not click.confirm("Do you want to continue?"):
                    logger.info("Migration cancelled by user")
                    return
            with FAILED_LOG_FILE_PATH.open("a") as f:
                json.dump(
                    {
                        "repo_id": repo_id,
                        "error": str(e),
                        "datetime": datetime.now().isoformat(),
                    },
                    f,
                )
                f.write("\n")


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



@dataclass
class ReadmeRegenerationTarget:
    repo_id: str
    pr_url: str


def get_pr_number_from_pr_url(pr_url: str) -> int:
    return int(re.search(r"/(\d+)$", pr_url).group(1))


def regenerate_readme_for_pr(
    hf_api: HfApi,
    anthropic_client: Anthropic,
    target: ReadmeRegenerationTarget,
    output_dir_path: str | None,
    working_dir_path: str | None,
    auto: bool,
    upload: bool,
):
    logger.info(f"Regenerating README.md for {target.repo_id}...")

    output_dir = Path(output_dir_path) if output_dir_path else None
    working_dir = Path(working_dir_path) if working_dir_path else None

    with temp_dir_if_none(output_dir) as output_dir, temp_dir_if_none(working_dir) as working_dir:
        repo_output_dir: Path = output_dir / target.repo_id
        if repo_output_dir.exists():
            logger.warning(f"Output directory {repo_output_dir} already exists. This script will overwrite its contents.")
        repo_output_dir.mkdir(parents=True, exist_ok=True)

        repo_info = hf_api.repo_info(target.repo_id)
        migrate_readme(hf_api=hf_api, anthropic_client=anthropic_client, model_info=repo_info, output_dir=repo_output_dir, auto=auto)

        if not upload:
            logger.info("Skipping upload")
            return

        if not auto:
            if not click.confirm(f"Are you sure you want to upload README.md to {target.repo_id} (PR #{get_pr_number_from_pr_url(target.pr_url)})?"):
                logger.info("Upload cancelled by user")
                return

        pr_number = get_pr_number_from_pr_url(target.pr_url)
        logger.info(f"Uploading README.md to {target.repo_id} (PR #{pr_number})...")
        hf_api.upload_file(
            path_or_fileobj=repo_output_dir / "README.md",
            path_in_repo="README.md",
            repo_id=target.repo_id,
            repo_type="model",
            revision=f"refs/pr/{pr_number}"
        )


@cli.command()
@click.option("--repo", required=False, multiple=True, type=str)
@click.option("--output-dir", required=False, type=click.Path(exists=False))
@click.option("--working-dir", required=False, type=click.Path(exists=False))
@click.option("--auto", required=False, is_flag=True)
@click.option("--upload", required=False, is_flag=True)
def regenerate_readme(
    repo: tuple[str],
    output_dir: str | None,
    working_dir: str | None,
    auto: bool,
    upload: bool,
):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set")

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    if auto and not upload:
        if not click.confirm("Are you sure you want to run in auto mode without uploading?"):
            logger.info("Migration cancelled by user")
            return

    hf_api = HfApi(token=token)
    anthropic_client = Anthropic(api_key=anthropic_api_key)

    targets: list[ReadmeRegenerationTarget] = []
    with LOG_FILE_PATH.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            current_log = json.loads(line)
            targets.append(ReadmeRegenerationTarget(repo_id=current_log["repo_id"], pr_url=current_log["pr_url"]))

    if repo:
        targets = [t for t in targets if t.repo_id in repo]

    logger.info(f"Regenerating README.md for {len(targets)} repos:\n{'\n'.join([' - ' + t.repo_id for t in targets])}")
    if not auto:
        if not click.confirm("Are you sure you want to regenerate README.md for these repos?"):
            logger.info("Regeneration cancelled by user")
            return

    for target in targets:
        logger.info(f"Regenerating README.md for {target.repo_id}...")
        regenerate_readme_for_pr(
            hf_api=hf_api,
            anthropic_client=anthropic_client,
            target=target,
            output_dir_path=output_dir,
            working_dir_path=working_dir,
            auto=auto,
            upload=upload,
        )


@cli.command()
@click.option("--repo", required=False, multiple=True, type=str)
@click.option("--author", required=False, type=str)
@click.option("--model-name", required=False, type=str)
@click.option("--task", required=False, type=str)
@click.option("--filter", required=False, multiple=True, type=str)
@click.option("--output-dir", required=False, type=click.Path(exists=False))
def preview_readme(
    repo: tuple[str],
    author: str | None,
    model_name: str | None,
    task: str | None,
    filter: tuple[str],
    output_dir: str | None,
):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set")

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    hf_api = HfApi(token=token)
    anthropic_client = Anthropic(api_key=anthropic_api_key)

    repo: list[str] = list(repo)

    if author or model_name or task or filter:
        search_results = hf_api.list_models(
            library="transformers.js",
            author=author,
            model_name=model_name,
            pipeline_tag=task,
            filter=filter,
        )
        searched_repo_ids = [r.id for r in search_results]
        repo = repo + searched_repo_ids

    logger.info(f"Previewing README.md for {len(repo)} repos:\n{'\n'.join([' - ' + r for r in repo])}")

    output_dir = Path(output_dir) if output_dir else None

    with temp_dir_if_none(output_dir) as output_dir:
        for repo_id in repo:
            logger.info(f"Previewing README.md for {repo_id}...")
            repo_info = hf_api.repo_info(repo_id)
            migrate_readme(
                hf_api=hf_api,
                anthropic_client=anthropic_client,
                model_info=repo_info,
                output_dir=output_dir,
                auto=False
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cli()
