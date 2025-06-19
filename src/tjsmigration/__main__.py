import os
import logging
from pathlib import Path

import click
from huggingface_hub import HfApi

from .model_migration import migrate_model_files
from .readme_migration import migrate_readme

logger = logging.getLogger(__name__)


@click.command()
@click.option("--repo-id", required=True)
@click.option("--output-dir", required=False, type=click.Path(exists=False))
@click.option("--upload", required=False, type=bool, default=False)
def migrate(repo_id: str, output_dir: str | None, upload: bool):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set")

    output_dir = Path(output_dir) if output_dir else None

    hf_api = HfApi(token=token)
    repo_info = hf_api.repo_info(repo_id)

    migrate_model_files(hf_api=hf_api, repo_id=repo_id, output_dir=output_dir, upload=upload)
    migrate_readme(hf_api=hf_api, repo_id=repo_id, output_dir=output_dir, upload=upload)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    migrate()
