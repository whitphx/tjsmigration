from pathlib import Path
import logging
import difflib
import subprocess
import tempfile
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from huggingface_hub import HfApi
from anthropic import Anthropic

logger = logging.getLogger(__name__)

from .tempdir import temp_dir_if_none
from .task_type import infer_transformers_task_type


class UserAction(Enum):
    ACCEPT = "accept"
    REGENERATE = "regenerate"
    QUIT = "quit"


@dataclass
class UserResponse:
    action: UserAction
    content: Optional[str] = None
    additional_instruction: Optional[str] = None


def print_colored_diff(original: str, new: str, filename: str = "README.md"):
    """Print a colored diff between original and new content."""
    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    print(f"\n{BLUE}=== DIFF for {filename} ==={RESET}")

    # Split content into lines for diff
    original_lines = original.splitlines()
    new_lines = new.splitlines()

    # Generate diff
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"{filename} (original)",
        tofile=f"{filename} (migrated)",
        lineterm=""
    )

    # Print colored diff
    for line in diff:
        if line.startswith('---') or line.startswith('+++'):
            print(f"{BLUE}{line}{RESET}")
        elif line.startswith('@@'):
            print(f"{BLUE}{line}{RESET}")
        elif line.startswith('+'):
            print(f"{GREEN}{line}{RESET}")
        elif line.startswith('-'):
            print(f"{RED}{line}{RESET}")
        else:
            print(line)

    print(f"{BLUE}=== END DIFF ==={RESET}\n")


def get_user_confirmation_and_edit(content: str, existing_instructions: list[str] = None) -> UserResponse:
    """Ask user for confirmation and optionally allow editing."""
    while True:
        print("\nOptions:")
        print("  [y] Proceed with the generated README")
        print("  [e] Edit the generated README in your default editor")
        print("  [r] Regenerate (return to regeneration)")
        print("  [q] Quit without saving")

        choice = input("\nWhat would you like to do? [y/e/r/q]: ").lower().strip()

        if choice == 'y':
            return UserResponse(action=UserAction.ACCEPT, content=content)
        elif choice == 'e':
            # Create a temporary file for editing
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()

                # Get the default editor
                editor = os.environ.get('EDITOR', 'nano')

                try:
                    # Open the file in the editor
                    subprocess.run([editor, tmp_file.name], check=True)

                    # Read the edited content
                    with open(tmp_file.name, 'r') as f:
                        edited_content = f.read()

                    # Show diff of the edited changes
                    print_colored_diff(content, edited_content, "README.md (after editing)")

                    return UserResponse(action=UserAction.ACCEPT, content=edited_content)
                except subprocess.CalledProcessError:
                    print(f"Error: Could not open editor '{editor}'. Please set the EDITOR environment variable.")
                    continue
                except FileNotFoundError:
                    print(f"Error: Editor '{editor}' not found. Please install it or set a different EDITOR.")
                    continue
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except OSError:
                        pass
        elif choice == 'r':
            # Show existing instructions if any
            if existing_instructions:
                print(f"\nExisting instructions: {'\n'.join(existing_instructions)}")

            additional_instruction = input("\nEnter additional instruction for regeneration (or press Enter for none): ").strip()
            return UserResponse(
                action=UserAction.REGENERATE,
                additional_instruction=additional_instruction if additional_instruction else None
            )
        elif choice == 'q':
            return UserResponse(action=UserAction.QUIT)
        else:
            print("Invalid choice. Please enter 'y', 'e', 'r', or 'q'.")


def update_readme_content(anthropic_client: Anthropic, content: str, task_type: str, repo_id: str, additional_instruction: str = "") -> str:
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

{"## ADDITIONAL USER INSTRUCTION:" if additional_instruction else ""}
{additional_instruction}

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


def migrate_readme(hf_api: HfApi, anthropic_client: Anthropic, repo_id: str, output_dir: Path | None, upload: bool):
    downloaded_path = hf_api.snapshot_download(repo_id=repo_id, repo_type="model")
    readme_path = Path(downloaded_path) / "README.md"

    with readme_path.open("r") as f:
        orig_readme_content = f.read()

    repo_info = hf_api.repo_info(repo_id)
    task_type = infer_transformers_task_type(repo_info)

    # Generate and potentially regenerate content based on user feedback
    additional_instructions = []
    while True:
        # Combine all additional instructions
        combined_instruction = "\n".join(additional_instructions) if additional_instructions else ""

        # Generate the new content
        new_readme_content = update_readme_content(anthropic_client, orig_readme_content, task_type, repo_id, combined_instruction)

        # Print colored diff
        print_colored_diff(orig_readme_content, new_readme_content)

        # Ask user for confirmation and allow editing
        user_response = get_user_confirmation_and_edit(new_readme_content, additional_instructions)

        if user_response.action == UserAction.ACCEPT:
            final_readme_content = user_response.content
            break
        elif user_response.action == UserAction.REGENERATE:
            if user_response.additional_instruction:
                additional_instructions.append(user_response.additional_instruction)
                print(f"Additional instructions: {'\n'.join(additional_instructions)}")
            print("\nRegenerating README content...")
            continue
        elif user_response.action == UserAction.QUIT:
            logger.info("Operation cancelled by user")
            return

    with temp_dir_if_none(output_dir) as output_dir:
        output_readme_path = output_dir / "README.md"
        with output_readme_path.open("w") as f:
            f.write(final_readme_content)

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
