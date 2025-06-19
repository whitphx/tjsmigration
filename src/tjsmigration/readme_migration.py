from pathlib import Path
import logging

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


def update_readme_content(readme_content: str) -> str:
    return readme_content.replace("onnx", "onnx_quantized")


def migrate_readme(hf_api: HfApi, repo_id: str, output_dir: Path, upload: bool):
    downloaded_path = hf_api.snapshot_download(repo_id=repo_id, repo_type="model")
    readme_path = Path(downloaded_path) / "README.md"

    with readme_path.open("r") as f:
        readme_content = f.read()

    # replace the content of the readme
    readme_content = update_readme_content(readme_content)

    output_readme_path = output_dir / "README.md"
    with output_readme_path.open("w") as f:
        f.write(readme_content)

    if upload:
        logger.info(f"Uploading README.md to {repo_id}...")
        hf_api.upload_file(
            path_or_fileobj=output_readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update README.md for Transformers.js v3",
            create_pr=True,
        )
        logger.info(f"Uploaded README.md to {repo_id}")
    else:
        logger.info("Skipping upload")
