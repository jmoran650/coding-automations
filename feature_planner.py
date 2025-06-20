#!/usr/bin/env python3
"""
feature_planner.py
──────────────────
Generate an ordered implementation plan (“task list”) for adding a new feature
to an existing code-base.

High-level flow
───────────────
1. Accept a natural-language **feature prompt** from the CLI.
2. Run *deep research* on that prompt using either **OpenAI** or **Google Gemini**
   (selectable via `--provider`).
3. Read the local code-base (recursively, with binary-file detection & smart
   ignore rules).
4. Send both the research report *and* the code-base to **Gemini 1.5 Pro** to
   obtain an ordered **task list**.
5. Print the tasks in a Codex-friendly numbered format.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import List, NoReturn

import pathspec

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types import GenerateContentResponse
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm

# --- Environment & API Configuration ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG = os.getenv("OPENAI_ORG")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Constants ---
_CODE_GATHER_EXCLUDE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "node_modules", "build", "dist"}
_TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".rst", ".json", ".toml",
    ".yaml", ".yml", ".html", ".css", ".scss", ".java", ".c", ".h", ".cpp", ".hpp",
}

# --- Data Structures ---
@dataclass
class ResearchReport:
    """Holds the output from the initial research phase."""
    provider: str
    content: str

@dataclass
class TaskList:
    """Holds the final list of implementation tasks."""
    tasks: List[str]

# --- Core Functions ---
def _fatal(msg: str, exit_code: int = 1) -> NoReturn:
    """Prints an error message to stderr and exits."""
    print(f"\033[91m❌ {msg}\033[0m", file=sys.stderr)
    sys.exit(exit_code)

def validate_env(provider: str) -> None:
    """Checks if the required API keys are present in the environment."""
    if provider == "openai" and not OPENAI_API_KEY:
        _fatal("OPENAI_API_KEY is required for provider 'openai'.")
    if not GOOGLE_API_KEY:
        _fatal("GOOGLE_API_KEY is required (Gemini is always used for the final plan).")

def init_sdks() -> None:
    """Configures the API clients with keys from the environment."""
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    # The OpenAI client is initialized on-demand in `run_research`.

def gather_codebase(root: pathlib.Path) -> str:
    """
    Recursively scans a directory for text files and concatenates their content.
    Respects .gitignore rules and a hardcoded set of excluded directories.
    """
    code_parts: List[str] = []
    
    # Load .gitignore rules if available
    gitignore_path = root / ".gitignore"
    spec_lines = gitignore_path.read_text().splitlines() if gitignore_path.exists() else []
    gitignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", spec_lines)

    paths_to_scan: List[pathlib.Path] = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune directories from the walk itself to improve efficiency
        # Also check directories against .gitignore
        current_path = pathlib.Path(dirpath)
        rel_dir_path = current_path.relative_to(root)

        # Exclude hardcoded dirs AND gitignored dirs from being traversed
        excluded_by_name = set(d for d in dirnames if d in _CODE_GATHER_EXCLUDE_DIRS)
        excluded_by_gitignore = set(d for d in dirnames if gitignore_spec.match_file(str(rel_dir_path / d)))
        
        dirnames[:] = [d for d in dirnames if d not in excluded_by_name and d not in excluded_by_gitignore]

        for filename in filenames:
            file_path = current_path / filename
            rel_file_path = file_path.relative_to(root)
            
            # Final check on the full file path
            if gitignore_spec.match_file(str(rel_file_path)):
                continue
            
            paths_to_scan.append(file_path)

    for path in tqdm(paths_to_scan, desc="📦 Scanning code-base", unit="file"):
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        try:
            # Simple binary file detection
            with open(path, "rb") as fh:
                if b"\0" in fh.read(512):
                    continue
            
            text = path.read_text(encoding="utf-8", errors="replace")
            rel_path = path.relative_to(root)
            code_parts.append(f"\n\n# ─── File: {rel_path} ───\n{text}")
        except Exception as exc:
            print(f"⚠️  Skipping {path.relative_to(root)} ({exc})", file=sys.stderr)

    return "\n".join(code_parts)


def run_research(prompt: str, provider: str, temperature: float = 0.7) -> ResearchReport:
    """Performs technical research using the specified AI provider."""
    print(f"🔬 Performing research with {provider.capitalize()}...")
    system_msg = (
        "You are an expert software architect. "
        "Perform a thorough technical & domain analysis of the following feature "
        "request. Include relevant libraries, APIs, edge cases, risks, and prior art."
    )

    if provider == "openai":
        try:
            client: OpenAI = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG)
            response: ChatCompletion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            _fatal(f"OpenAI API call failed: {e}")
    else:
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            gemini_resp: GenerateContentResponse = model.generate_content(
                [system_msg, prompt],
                generation_config=genai.GenerationConfig(
                    temperature=temperature, max_output_tokens=2048
                ),
            )
            content = gemini_resp.text
        except Exception as e:
            _fatal(f"Google Gemini API call failed: {e}")

    return ResearchReport(provider=provider, content=content.strip())

def generate_task_list(report: ResearchReport, codebase: str) -> TaskList:
    """Generates the final implementation plan using Gemini 1.5 Pro."""
    print("\n🤔 Generating implementation plan with Gemini 1.5 Pro...")
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

    system_prompt = (
        "You are a senior software engineer creating a step-by-step implementation plan.\n"
        "Based on the provided research report and the existing codebase, generate a "
        "numbered list of tasks to implement the requested feature.\n"
        "Focus on a clear, ordered sequence of actions. For each task, be specific about "
        "the file(s) that need to be created or modified.\n"
        "Your output must ONLY be the numbered list of tasks and nothing else."
    )

    user_content = (
        f"## RESEARCH REPORT (from {report.provider}) ##\n\n"
        f"{report.content}\n\n"
        f"## EXISTING CODEBASE ##\n\n"
        f"{codebase}"
    )

    try:
        response: GenerateContentResponse = model.generate_content(
            [system_prompt, user_content],
            # Lower temperature for more deterministic, focused planning
            generation_config=genai.GenerationConfig(temperature=0.2),
        )
        # Basic parsing: split by newline and filter for lines starting with a digit.
        tasks = [
            line.strip() for line in response.text.split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]
        return TaskList(tasks=tasks)
    except Exception as e:
        _fatal(f"Gemini task generation failed: {e}")


def main():
    """Main script execution."""
    cli = argparse.ArgumentParser(
        description="Generate an AI-assisted implementation plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_argument("prompt", help="A natural-language description of the feature.")
    cli.add_argument(
        "-p", "--provider", choices=["openai", "gemini"], default="gemini",
        help="The provider to use for the initial research phase."
    )
    cli.add_argument(
        "--root", type=pathlib.Path, default=pathlib.Path.cwd(),
        help="The root directory of the codebase to scan."
    )
    args = cli.parse_args()

    validate_env(args.provider)
    init_sdks()

    # Steps 1 & 2: Run deep research
    research_report = run_research(args.prompt, args.provider)

    # Step 3: Gather codebase
    codebase_str = gather_codebase(args.root)

    # Step 4: Generate the final task list
    task_list = generate_task_list(research_report, codebase_str)

    # Step 5: Print the final output
    print("\n\n✨ Implementation Plan ✨\n")
    if not task_list.tasks:
        print("Could not generate a task list. The model may have returned an empty response.")
    else:
        # Re-number the tasks to ensure a clean, consistent list.
        for i, task in enumerate(task_list.tasks, 1):
            # Strip any leading numbering from the model's output to avoid duplication.
            clean_task = task.lstrip("0123456789. ")
            print(f"{i}. {clean_task}")
    print()

if __name__ == "__main__":
    main()