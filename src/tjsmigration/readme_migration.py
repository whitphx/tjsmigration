from pathlib import Path
import logging

from huggingface_hub import HfApi
from anthropic import Anthropic

logger = logging.getLogger(__name__)

from .task_type import infer_transformers_task_type


def update_readme_content(anthropic_client: Anthropic, content: str, task_type: str, repo_id: str) -> str:
    prompt = f"""You are migrating a Transformers.js model repository README from v2 to v3. Your task is to update the README content while preserving its original structure and purpose.

## CRITICAL REQUIREMENTS:
1. **Output only the migrated README content** - NO wrapper text, explanations, or meta-commentary
2. **Preserve original structure** - Keep the same sections, formatting, and overall organization
3. **Minimal changes only** - Only update what's necessary for v3 compatibility
4. **PRESERVE FRONTMATTER** - Keep all YAML frontmatter (content between --- lines) exactly as-is

## Required Changes:
1. **Package name**: Change `@xenova/transformers` to `@huggingface/transformers`
2. **Installation instructions**: If there is no installation instructions, add it with the template below. If it already exists, update it to use the new package name.
3. **Add basic usage example**: If no code examples exist, add a basic usage example. If there are code examples, update them to use the new package name and signatures as follows.
4. **Remove inline install comments**: Remove `// npm i @xenova/transformers` comments from code blocks because the installation instructions are already added as above
5. **Modern JavaScript**: Use `const` instead of `let` or `var` for variables that aren't reassigned
6. **Add semicolons**: Ensure statements end with semicolons where appropriate
7. **Keep code formats**: Keep the code formats such as white spaces, line breaks, etc. as is
8. **Add the third argument to the pipeline function**: Add the third argument to the pipeline function that is a configuration object whose value is {{ dtype: "fp32" }}. It should be in the next line after the pipeline creation line with a comment saying '// Options: "fp32", "fp16", "q8", "q4"'

## Installation Section Template:
When adding installation instructions, use this format before the first code example:

'''''
If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```
'''''

## Basic Usage Example Template:
Add or update a basic usage example based on the model type. Use the repository ID from the prompt to create an appropriate example:

```js
import {{ pipeline }} from '@huggingface/transformers';

// Create the pipeline
const pipe = await pipeline('{task_type}', '{repo_id}');

// Use the model
const result = await pipe('input text or data');
console.log(result);
```

## STRICT GUIDELINES:
- **NEVER remove frontmatter** - Keep all YAML metadata between --- lines exactly as-is
- **ADD installation instructions** - Always add them before code examples if missing
- **ADD basic usage example** - If no code examples exist, add a simple usage example based on the model type
- DO NOT add explanatory text about what the code does beyond basic usage
- DO NOT move example outputs or change code structure
- DO NOT add sections that weren't in the original (except installation and basic usage)
- DO NOT add wrapper text like "Here is the migrated content"
- PRESERVE comments that are example outputs (like "// Found car at...")
- Keep the exact same markdown structure and sections
- Return ONLY the migrated README content, nothing else


## Original README Content:
{content}

## MIGRATED README (output only this):"""

    response = anthropic_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=4000,
        temperature=0.1,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    migrated_content = response.content[0].text.strip()
    return migrated_content


def migrate_readme(hf_api: HfApi, anthropic_client: Anthropic, repo_id: str, output_dir: Path, upload: bool):
    downloaded_path = hf_api.snapshot_download(repo_id=repo_id, repo_type="model")
    readme_path = Path(downloaded_path) / "README.md"

    with readme_path.open("r") as f:
        readme_content = f.read()

    repo_info = hf_api.repo_info(repo_id)
    task_type = infer_transformers_task_type(repo_info)

    # replace the content of the readme
    readme_content = update_readme_content(anthropic_client, readme_content, task_type, repo_id)
    logger.info(f"MIGRATED README: {readme_content}")

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
