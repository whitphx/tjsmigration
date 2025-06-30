from pathlib import Path
import logging
import difflib
import subprocess
import tempfile
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import click
from huggingface_hub import HfApi, ModelInfo
from anthropic import Anthropic

from .task_type import infer_transformers_task_type


logger = logging.getLogger(__name__)


class UserAction(Enum):
    ACCEPT = "accept"
    EDIT = "edit"
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

        choice = click.prompt(
            "What would you like to do? [y/e/r/q]: ",
            type=click.Choice(['y', 'e', 'r', 'q']),
        )

        if choice == 'y':
            return UserResponse(action=UserAction.ACCEPT)
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

                    return UserResponse(action=UserAction.EDIT, content=edited_content)
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


def _get_usage_example(task_type: str, repo_id: str) -> tuple[str, str | None]:
    if task_type == "fill-mask":
        return f"""const unmasker = await pipeline('fill-mask', '{repo_id}');
const output = await unmasker('The goal of life is [MASK].');
""", 'Perform masked language modelling (a.k.a. "fill-mask")'
    elif task_type == "question-answering":
        return f"""const answerer = await pipeline('question-answering', '{repo_id}');
const question = 'Who was Jim Henson?';
const context = 'Jim Henson was a nice puppet.';
const output = await answerer(question, context);
""", 'Run question answering'
    elif task_type == "summarization":
        return f"""const generator = await pipeline('summarization', '{repo_id}');
const text = 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, ' +
  'and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. ' +
  'During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest ' +
  'man-made structure in the world, a title it held for 41 years until the Chrysler Building in New ' +
  'York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to ' +
  'the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the ' +
  'Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second ' +
  'tallest free-standing structure in France after the Millau Viaduct.';
const output = await generator(text, {{
  max_new_tokens: 100,
}});
""", 'Summarization'
    elif task_type == "sentiment-analysis" or task_type == "text-classification":
        return f"""const classifier = await pipeline('{task_type}', '{repo_id}');
const output = await classifier('I love transformers!');
""", None
    elif task_type == "text-generation":
        return f"""const generator = await pipeline('text-generation', '{repo_id}');
const output = await generator('Once upon a time, there was', {{ max_new_tokens: 10 }});
""", 'Text generation'
    elif task_type == "text2text-generation":
        return f"""const generator = await pipeline('text2text-generation', '{repo_id}');
const output = await generator('how can I become more healthy?', {{
  max_new_tokens: 100,
}});
""", 'Text-to-text generation'
    elif task_type == "token-classification" or task_type == "ner":
        return f"""const classifier = await pipeline('token-classification', '{repo_id}');
const output = await classifier('My name is Sarah and I live in London');
""", 'Perform named entity recognition'
    elif task_type == "translation":
        return f"""const translator = await pipeline('translation', '{repo_id}');
const output = await translator('Life is like a box of chocolate.', {{
  src_lang: '...',
  tgt_lang: '...',
}});
""", 'Multilingual translation'
    elif task_type == "zero-shot-classification":
        return f"""const classifier = await pipeline('zero-shot-classification', '{repo_id}');
const text = 'Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.';
const labels = [ 'mobile', 'billing', 'website', 'account access' ];
const output = await classifier(text, labels);
""", 'Zero shot classification'
    elif task_type == "feature-extraction":
        return f"""const extractor = await pipeline('feature-extraction', '{repo_id}');
const output = await extractor('This is a simple test.');
""", 'Run feature extraction'
# Vision
    elif task_type == "background-removal":
        return f"""const segmenter = await pipeline('background-removal', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/portrait-of-woman_small.jpg';
const output = await segmenter(url);
""", 'Perform background removal'
    elif task_type == "depth-estimation":
        return f"""const depth_estimator = await pipeline('depth-estimation', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const out = await depth_estimator(url);
""", 'Depth estimation'
    elif task_type == "image-classification":
        return f"""const classifier = await pipeline('image-classification', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await classifier(url);
""", 'Classify an image'
    elif task_type == "image-segmentation":
        return f"""const segmenter = await pipeline('image-segmentation', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await segmenter(url);
""", 'Perform image segmentation'
    elif task_type == "image-to-image":
        return f"""const processor = await pipeline('image-to-image', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await processor(url);
""", None
    elif task_type == "object-detection":
        return f"""const detector = await pipeline('object-detection', '{repo_id}');
const img = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await detector(img, {{ threshold: 0.9 }});
""", 'Run object-detection'
    elif task_type == "image-feature-extraction":
        return f"""const image_feature_extractor = await pipeline('image-feature-extraction', '{repo_id}');
const url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png';
const features = await image_feature_extractor(url);
""", 'Perform image feature extraction'
# Audio
    elif task_type == "audio-classification":
        return f"""const classifier = await pipeline('audio-classification', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const output = await classifier(url);
""", 'Perform audio classification'
    elif task_type == "automatic-speech-recognition":
        return f"""const transcriber = await pipeline('automatic-speech-recognition', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const output = await transcriber(url);
""", 'Transcribe audio from a URL'
    elif task_type == "text-to-audio" or task_type == "text-to-speech":
        return f"""const synthesizer = await pipeline('text-to-speech', '{repo_id}');
const output = await synthesizer('Hello, my dog is cute');
""", 'Generate audio from text'
# Multimodal
    elif task_type == "document-question-answering":
        return f"""const qa_pipeline = await pipeline('document-question-answering', '{repo_id}');
const image = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/invoice.png';
const question = 'What is the invoice number?';
const output = await qa_pipeline(image, question);
""", 'Answer questions about a document'
    elif task_type == "image-to-text":
        return f"""const captioner = await pipeline('image-to-text', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await captioner(url);
""", 'Generate a caption for an image'
    elif task_type == "zero-shot-audio-classification":
        return f"""const classifier = await pipeline('zero-shot-audio-classification', '{repo_id}');
const audio = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/dog_barking.wav';
const candidate_labels = ['dog', 'vaccum cleaner'];
const scores = await classifier(audio, candidate_labels);
""", 'Perform zero-shot audio classification'
    elif task_type == "zero-shot-image-classification":
        return f"""const classifier = await pipeline('zero-shot-image-classification', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await classifier(url, ['tiger', 'horse', 'dog']);
""", 'Zero shot image classification'
    elif task_type == "zero-shot-object-detection":
        return f"""const detector = await pipeline('zero-shot-object-detection', '{repo_id}');
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/astronaut.png';
const candidate_labels = ['human face', 'rocket', 'helmet', 'american flag'];
const output = await detector(url, candidate_labels);
""", 'Zero-shot object detection'
    else:
        logger.warning(f"No usage example found for task type: {task_type}")
        return f"""const pipe = await pipeline('{task_type}', '{repo_id}');
const result = await pipe('input text or data');
console.log(result);
""", None


EXAMPLE_OUTPUT = """
---
base_model: openai/whisper-medium
library_name: transformers.js
---
https://huggingface.co/openai/whisper-medium with ONNX weights to be compatible with Transformers.js.

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
<installation_instructions>
```

**Example:** <example_description if specified>

```js
<example_code>
```

Note: Having a separate repo for ONNX weights is intended to be a temporary solution until WebML gains more traction. If you would like to make your models web-ready, we recommend converting to ONNX using [🤗 Optimum](https://huggingface.co/docs/optimum/index) and structuring your repo like this one (with ONNX weights located in a subfolder named `onnx`).
"""


def _generate_prompt(content: str, task_type: str, repo_id: str, additional_instructions: list[str]) -> str:
    usage_example, usage_example_description = _get_usage_example(task_type, repo_id)
    if usage_example_description is None:
        usage_example_description = ""

    prompt = f"""You are migrating a Transformers.js model repository README from v2 to v3. Your task is to update the README content while preserving its original structure and purpose.

## CRITICAL REQUIREMENTS:
1. **Output only the migrated README content** - NO wrapper text, explanations, or meta-commentary
2. **Preserve original structure** - Keep the same sections, formatting, and overall organization. When adding the installation instructions, add it in the same section as the basic usage example, unless the original README has different structure.
3. **Minimal changes only** - Only update what's necessary for v3 compatibility
4. **PRESERVE FRONTMATTER** - Keep all YAML frontmatter (content between --- lines) exactly as-is

## Required Changes:
1. **Package name**: Change `@xenova/transformers` to `@huggingface/transformers`
2. **"Usage" section**: Add a new section with heading "## Usage (Transformers.js)" containing installation instructions and basic usage example unless it already exists.
3. **Installation instructions**: Add an installation instruction based on the template below unless it already exists. If it already exists, update it to use the new package name. The installation instructions should be added in the "Usage" section.
4. **Add basic usage example**: Add a basic usage example based on the template below unless it already exists. If there are code examples, update them to use the new package name and signatures as follows. The basic usage example should be added in the "Usage" section.
5. **Remove inline install comments**: Remove `// npm i @xenova/transformers` comments from code blocks because the installation instructions are already added as above
6. **Replace `{{ quantized: false }}` to `{{ dtype: "fp32" }}` and `{{ topk: ... }}` to `{{ top_k: ... }}`**: The `quantized` option is no longer supported in Transformers.js. The `topk` option is now renamed to `top_k`.
7. **Modern JavaScript**: Use `const` instead of `let` or `var` for variables that aren't reassigned
8. **Add semicolons**: Ensure statements end with semicolons where appropriate
9. **Keep code formats**: Keep the code formats such as white spaces, line breaks, etc. as is

## Installation Section Template:
When adding installation instructions, use this format before the first code example:

'''''
If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```
'''''

## Basic Usage Example Template:
Add or update a basic usage example as the template below.

'''''
**Example:** {usage_example_description}.

```js
import {{ pipeline }} from '@huggingface/transformers';

{usage_example}
```
'''''

#### GUIDELINES for the basic usage example:
- The "Example" is not a heading. Insert it as well in the "Usage" section.
- Use the task name and model name passed to the pipeline function as-is from this template. DO NOT change them.
- The original README content may include different model names as references, but DO NOT use them in the usage example.
- Replace the elements such as the model name, task type, pipeline variable name in the existing examples.

## Example Outputs:

'''''
{EXAMPLE_OUTPUT}
'''''

## Properties of the model:
ID: {repo_id}
Task type: {task_type}

## STRICT GUIDELINES:
- **NEVER remove frontmatter** - Keep all YAML metadata between --- lines exactly as-is
- **ADD installation instructions** - Always add them before code examples if missing
- **ADD basic usage example** - If no code examples exist, add a simple usage example based on the model type
- DO NOT add explanatory text about what the code does beyond basic usage
- DO NOT move example outputs or change code structure
- DO NOT change the task name and model name passed to the pipeline function from the template above, that are the task type and model name.
- DO NOT add sections that weren't in the original (except installation and basic usage)
- DO NOT add wrapper text like "Here is the migrated content"
- PRESERVE comments that are aligned with the sample code semantics (like "// Create a pipeline").
- PRESERVE comments that are example outputs (like "// Found car at...")
- Keep the exact same markdown structure and sections. No need to add new headings if the original README has no headings.
- DO NOT change the markdown structure, move the existing headings, or delete any existing elements such as sections, dividers, etc.
- Return ONLY the migrated README content, nothing else

{"## ADDITIONAL USER INSTRUCTION:" if additional_instructions else ""}
{'\n\n'.join(additional_instructions)}

## Original README Content:
{content}

## MIGRATED README (output only this):"""
    return prompt


def call_readme_update_llm(anthropic_client: Anthropic, orig_content: str, task_type: str, repo_id: str, additional_instructions: list[str]) -> str:
        prompt = _generate_prompt(orig_content, task_type, repo_id, additional_instructions)

        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=4000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        proposed_content = response.content[0].text.strip()
        return proposed_content

def update_readme_content(anthropic_client: Anthropic, orig_content: str, task_type: str, repo_id: str, auto: bool) -> str:
    additional_instructions = []
    while True:
        proposed_content = call_readme_update_llm(anthropic_client, orig_content, task_type, repo_id, additional_instructions)

        print_colored_diff(orig_content, proposed_content)

        if auto:
            return proposed_content
        else:
            user_response = get_user_confirmation_and_edit(proposed_content, additional_instructions)

        if user_response.action == UserAction.ACCEPT:
            return proposed_content
        elif user_response.action == UserAction.EDIT:
            return user_response.content
        elif user_response.action == UserAction.REGENERATE:
            additional_instructions.append(user_response.additional_instruction)
            continue
        elif user_response.action == UserAction.QUIT:
            logger.info("Operation cancelled by user")
            raise KeyboardInterrupt


def migrate_readme(
    hf_api: HfApi,
    anthropic_client: Anthropic,
    model_info: ModelInfo,
    output_dir: Path,
    auto: bool,
):
    repo_id = model_info.id

    downloaded_readme_path = hf_api.hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename="README.md",
    )
    downloaded_readme_path = Path(downloaded_readme_path)

    with downloaded_readme_path.open("r") as f:
        orig_readme_content = f.read()

    task_type = infer_transformers_task_type(model_info=model_info)

    new_readme_content = update_readme_content(anthropic_client, orig_readme_content, task_type, repo_id, auto)

    output_readme_path = output_dir / "README.md"
    with output_readme_path.open("w") as f:
        f.write(new_readme_content)
