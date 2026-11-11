"""
Prints the prompt the evaluation sends to a VLM, built by the same code the
runner uses (prompts.build_prompt and Runner._messages), so what you read is
what the model gets. Nothing is sent and no API key is needed.

With no --task it shows one question of every (task, question type), so each
distinct prompt is covered. The image goes first in the user message and is
shown as a placeholder (its bytes would drown the text); --save-image writes
the exact image bytes that would be sent.

Usage:
    uv run src/weedvlm/scripts/show-prompt.py --config test-run-configurations/claude-haiku-test.yaml
    uv run src/weedvlm/scripts/show-prompt.py --config ... --task density --index 3
    uv run src/weedvlm/scripts/show-prompt.py --config ... --question-id "CropAndWeed:4225:..."
    uv run src/weedvlm/scripts/show-prompt.py --config ... --task localisation --json
    uv run src/weedvlm/scripts/show-prompt.py --config ... --save-image /tmp/prompt-images
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from weedvlm.evaluation.config import load_evaluation_config, resolve_config_path
from weedvlm.evaluation.prompts import PROMPT_VERSION, build_prompt, density_rules_text
from weedvlm.evaluation.questions import open_benchmark
from weedvlm.evaluation.runner import Runner, _read_image


def _request_body(config, model, messages) -> dict:
    # Mirrors OpenRouterClient.chat, which builds the body inside the HTTP call.
    return {
        "model": model.id,
        "messages": messages,
        "temperature": config.temperature_for(model),
        "max_tokens": config.max_tokens_for(model),
        "usage": {"include": True},
        **model.extra_body,
    }


def _elide(messages: list[dict], placeholder: str) -> list[dict]:
    shown = json.loads(json.dumps(messages))
    for message in shown:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "image_url":
                    part["image_url"]["url"] = placeholder
    return shown


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: evaluation.example.yaml)")
    parser.add_argument("--task", default=None, help="Only this task, e.g. species_id, density, localisation")
    parser.add_argument("--index", type=int, default=0, help="Which question of each group to show (default 0)")
    parser.add_argument("--question-id", default=None, help="Show this exact question")
    parser.add_argument("--model", default=None, help="Model id, for its temperature/max_tokens/extra_body (default: first in config)")
    parser.add_argument("--json", action="store_true", help="Print the request body as JSON instead of plain text")
    parser.add_argument("--save-image", type=Path, default=None, help="Directory to write the image(s) that would be sent")
    args = parser.parse_args()

    try:
        config = load_evaluation_config(resolve_config_path(args.config))
        models = {m.id: m for m in config.models}
        model = models[args.model] if args.model else config.models[0]
        benchmark = open_benchmark(config)
        density_rules = density_rules_text(benchmark.density_config()) if config.tasks.density else None

        groups: dict[tuple[str, str], list] = {}
        for task, question in benchmark.questions:
            if args.question_id and question.id != args.question_id:
                continue
            if args.task and task != args.task:
                continue
            groups.setdefault((task, question.question_type.value), []).append(question)
        if not groups:
            print("error: no matching questions", file=sys.stderr)
            return 2

        for (task, question_type), questions in groups.items():
            if args.index >= len(questions):
                print(f"error: --index {args.index} but {task}/{question_type} has {len(questions)} questions", file=sys.stderr)
                return 2
            question = questions[args.index]
            prompt = build_prompt(question, density_rules=density_rules)
            data_url, sha256 = _read_image(benchmark, question)
            messages = Runner._messages(prompt, data_url)
            mime, size = data_url.split(";")[0][5:], len(benchmark.read_image(question))
            placeholder = f"data:{mime};base64,<{size} bytes elided, sha256 {sha256[:12]}>"

            if args.save_image:
                args.save_image.mkdir(parents=True, exist_ok=True)
                name = f"{task}-{question_type}-{args.index}{Path(question.image_path).suffix}"
                (args.save_image / name).write_bytes(benchmark.read_image(question))

            print(f"{'=' * 78}\n{task} / {question_type}  (prompt version {PROMPT_VERSION})\nquestion: {question.id}")
            if args.json:
                print(json.dumps(_request_body(config, model, _elide(messages, placeholder)), indent=2))
                continue
            print(
                f"model: {model.id}  temperature: {config.temperature_for(model)}  "
                f"max_tokens: {config.max_tokens_for(model)}  extra_body: {model.extra_body}\n"
                f"image: {question.image_path} ({mime}, {size} bytes)\n"
            )
            print(f"----- system message -----\n{prompt.system}\n")
            print(f"----- user message: [1] image (placeholder), [2] text -----\n{placeholder}\n")
            print(prompt.user_text)
            print(f"\n(required reply fields: {', '.join(prompt.required_fields)})\n")
    except (ValidationError, FileNotFoundError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
