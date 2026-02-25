"""
==============================================================================
playbook_retriever.py
==============================================================================

RLM-based playbook retriever for filtering large playbooks to relevant bullets.
Uses RLM (Recursive Language Models) to programmatically analyze the playbook
and task instruction to retrieve the most relevant guidelines.
"""

import json
import re
import sys
import os
from typing import Any, Literal

# Add RLM to path if not already installed
RLM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rlm")
if RLM_PATH not in sys.path:
    sys.path.insert(0, RLM_PATH)

from rlm import RLM
from rlm.logger import RLMLogger


# Core bullet IDs that should always be included (critical rules)
DEFAULT_CORE_BULLET_IDS = [
    "shr-00001",  # Code block formatting
    "shr-00005",  # Always look at API specs before calling
    "shr-00006",  # Write small chunks of code
    "shr-00021",  # Dynamic date calculations for time-sensitive operations
]

# Backend presets for easy switching
# Each preset defines: (rlm_backend, backend_kwargs)
BACKEND_PRESETS: dict[str, tuple[str, dict[str, Any]]] = {
    "openai": (
        "openai",
        {
            "model_name": "gpt-4o-mini",
            "temperature": 0,
        },
    ),
    "sambanova": (
        "openai",  # SambaNova uses OpenAI-compatible API
        {
            "model_name": "DeepSeek-V3.1",
            "base_url": "https://api.sambanova.ai/v1",
            "api_key_env_var": "SAMBANOVA_API_KEY",  # Will be resolved to actual key
            "temperature": 0,
        },
    ),
    "anthropic": (
        "anthropic",
        {
            "model_name": "claude-3-haiku-20240307",
            "temperature": 0,
        },
    ),
}

# Type for backend preset names
BackendPreset = Literal["openai", "sambanova", "anthropic"]


def get_backend_config(
    backend_preset: BackendPreset | None = None,
    backend: str | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Get backend configuration from preset or explicit settings.

    Args:
        backend_preset: Name of a predefined backend preset ("openai", "sambanova", "anthropic").
        backend: Explicit RLM backend name (overrides preset).
        backend_kwargs: Explicit backend kwargs (overrides preset).

    Returns:
        Tuple of (backend_name, backend_kwargs).

    Priority:
        1. If backend_kwargs is provided, use it with backend (or preset's backend)
        2. If backend_preset is provided, use the preset
        3. Default to "openai" preset
    """
    # Start with default preset
    default_backend, default_kwargs = BACKEND_PRESETS["openai"]

    if backend_preset and backend_preset in BACKEND_PRESETS:
        preset_backend, preset_kwargs = BACKEND_PRESETS[backend_preset]
        default_backend = preset_backend
        default_kwargs = preset_kwargs.copy()

    # Override with explicit settings
    final_backend = backend if backend else default_backend
    final_kwargs = backend_kwargs if backend_kwargs else default_kwargs.copy()

    # Resolve API key environment variable if specified
    if "api_key_env_var" in final_kwargs:
        env_var = final_kwargs.pop("api_key_env_var")
        api_key = os.environ.get(env_var)
        if api_key:
            final_kwargs["api_key"] = api_key
        else:
            print(f"[PlaybookRetriever] Warning: {env_var} not set in environment")

    return final_backend, final_kwargs


# Default path to the retriever system prompt file
DEFAULT_RETRIEVER_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "prompts", "playbook_retriever_prompt.txt"
)


def load_retriever_prompt(prompt_file_path: str | None = None) -> str:
    """Load the retriever system prompt from a file.

    Args:
        prompt_file_path: Path to the prompt file. If None, uses default path.

    Returns:
        The prompt text.
    """
    path = prompt_file_path or DEFAULT_RETRIEVER_PROMPT_PATH
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_playbook_bullets(playbook_text: str) -> list[dict]:
    """Parse playbook text into a list of bullet dictionaries.

    Returns:
        List of dicts with keys: id, content, section, helpful, harmful
    """
    bullets = []
    current_section = "general"

    # Patterns for parsing
    section_pattern = r'^##\s*(.+)$'
    bullet_pattern_full = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    bullet_pattern_simple = r'\[([^\]]+)\]\s*(.*)'

    for line in playbook_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check for section header
        section_match = re.match(section_pattern, line)
        if section_match:
            current_section = section_match.group(1).strip()
            continue

        # Check for bullet (full format with counts)
        match = re.match(bullet_pattern_full, line)
        if match:
            bullets.append({
                'id': match.group(1),
                'helpful': int(match.group(2)),
                'harmful': int(match.group(3)),
                'content': match.group(4),
                'section': current_section,
            })
            continue

        # Check for bullet (simple format)
        match = re.match(bullet_pattern_simple, line)
        if match:
            bullets.append({
                'id': match.group(1),
                'helpful': 0,
                'harmful': 0,
                'content': match.group(2).strip(),
                'section': current_section,
            })

    return bullets


def format_bullets_as_playbook(bullets: list[dict], original_playbook: str = None) -> str:
    """Format a list of bullet dicts back into playbook text.

    Groups bullets by section and formats them properly.
    """
    # Group bullets by section
    sections = {}
    for bullet in bullets:
        section = bullet.get('section', 'general')
        if section not in sections:
            sections[section] = []
        sections[section].append(bullet)

    # Build output text
    lines = []
    for section, section_bullets in sections.items():
        if section != "general":
            lines.append(f"## {section}")
        for bullet in section_bullets:
            lines.append(f"[{bullet['id']}] {bullet['content']}")
        lines.append("")  # Empty line between sections

    return '\n'.join(lines).strip()


class PlaybookRetriever:
    """RLM-based playbook retriever for filtering large playbooks to relevant bullets."""

    def __init__(
        self,
        backend_preset: BackendPreset | None = None,
        backend: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        max_bullets: int = 20,
        core_bullet_ids: list[str] | None = None,
        max_iterations: int = 15,
        verbose: bool = False,
        log_dir: str | None = None,
        prompt_file_path: str | None = None,
    ):
        """Initialize the PlaybookRetriever.

        Args:
            backend_preset: Predefined backend preset ("openai", "sambanova", "anthropic").
                          - "openai": Uses gpt-4o-mini (default)
                          - "sambanova": Uses DeepSeek-V3-0324 via SambaNova Cloud API
                          - "anthropic": Uses claude-3-haiku
            backend: RLM backend to use (overrides preset). Options: openai, anthropic, litellm, etc.
            backend_kwargs: Configuration for the backend (overrides preset).
                          For OpenAI: model_name, temperature, base_url, api_key
                          For SambaNova: uses OpenAI backend with custom base_url
            max_bullets: Maximum number of bullets to retrieve (excluding core bullets)
            core_bullet_ids: List of bullet IDs that should always be included.
                           Defaults to DEFAULT_CORE_BULLET_IDS.
            max_iterations: Maximum RLM iterations before timeout.
            verbose: Whether to print verbose output during retrieval.
            log_dir: Directory to save RLM execution logs (JSON-Lines format).
                    If provided, logs all iterations with prompts, responses,
                    and executed code to files named:
                    playbook_retriever_{uuid}.jsonl
            prompt_file_path: Path to the system prompt file. If None, uses default
                            at experiments/prompts/playbook_retriever_prompt.txt

        Examples:
            # Use OpenAI (default)
            retriever = PlaybookRetriever()

            # Use SambaNova with DeepSeek-V3
            retriever = PlaybookRetriever(backend_preset="sambanova")

            # Use custom model
            retriever = PlaybookRetriever(
                backend="openai",
                backend_kwargs={"model_name": "gpt-4o", "temperature": 0}
            )
        """
        # Resolve backend configuration from preset or explicit settings
        self.backend, self.backend_kwargs = get_backend_config(
            backend_preset=backend_preset,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )
        self.backend_preset = backend_preset
        self.max_bullets = max_bullets
        self.core_bullet_ids = core_bullet_ids if core_bullet_ids is not None else DEFAULT_CORE_BULLET_IDS
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.log_dir = log_dir
        self.system_prompt = load_retriever_prompt(prompt_file_path)

        if self.verbose:
            print(f"[PlaybookRetriever] Backend: {self.backend}")
            print(f"[PlaybookRetriever] Model: {self.backend_kwargs.get('model_name', 'unknown')}")
            if 'base_url' in self.backend_kwargs:
                print(f"[PlaybookRetriever] Base URL: {self.backend_kwargs['base_url']}")

    def retrieve(
        self,
        task_instruction: str,
        playbook_text: str,
        app_descriptions: dict[str, str] | None = None,
    ) -> str:
        """Retrieve relevant playbook bullets for a given task.

        Args:
            task_instruction: The task instruction to analyze.
            playbook_text: The full playbook text.
            app_descriptions: Dictionary of available apps and their descriptions.

        Returns:
            Filtered playbook text containing only relevant bullets.
        """
        # Track selection metadata for logging
        self._selection_metadata = {
            "task_instruction": task_instruction,
            "total_bullets": 0,
            "rlm_selected_ids": [],
            "rlm_raw_response": "",
            "parse_success": False,
            "used_fallback": False,
            "final_selected_ids": [],
            "final_bullet_count": 0,
        }

        # Parse all bullets from playbook
        all_bullets = parse_playbook_bullets(playbook_text)
        self._selection_metadata["total_bullets"] = len(all_bullets)

        if not all_bullets:
            print("[PlaybookRetriever] Warning: No bullets found in playbook, returning original")
            return playbook_text

        # Build bullet lookup
        bullet_lookup = {b['id']: b for b in all_bullets}

        # Create compact bullet representation (truncate content to reduce context size)
        # This reduces ~117KB playbook to ~20-30KB
        compact_bullets = []
        for b in all_bullets:
            content = b['content']
            # Truncate long content but keep enough for semantic understanding
            if len(content) > 200:
                content = content[:200] + "..."
            compact_bullets.append({
                'id': b['id'],
                'section': b['section'],
                'content': content,
            })

        # Prepare context for RLM (use compact representation instead of full playbook)
        context = {
            "task_instruction": task_instruction,
            "bullets": compact_bullets,  # Compact format instead of full playbook
            "app_descriptions": app_descriptions or {},
            "max_bullets": self.max_bullets,
            "core_bullet_ids": self.core_bullet_ids,
            "bullet_count": len(all_bullets),
        }

        if self.verbose:
            context_size = len(str(context))
            print(f"[PlaybookRetriever] Context size: {context_size} chars (reduced from {len(playbook_text)})")

        # Create logger if log_dir is specified
        logger = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            logger = RLMLogger(log_dir=self.log_dir, file_name="playbook_retriever")
            if self.verbose:
                print(f"[PlaybookRetriever] Logging to: {self.log_dir}")

        # Create RLM instance
        rlm = RLM(
            backend=self.backend,
            backend_kwargs=self.backend_kwargs,
            environment="local",
            max_iterations=self.max_iterations,
            custom_system_prompt=self.system_prompt,
            verbose=self.verbose,
            logger=logger,
        )

        # Build the root prompt (shown to the model)
        root_prompt = f"""Select the {self.max_bullets} most relevant playbook bullets for this task:

Task: {task_instruction}

Available apps: {list(app_descriptions.keys()) if app_descriptions else 'Not specified'}

Return a JSON list of bullet IDs. Core bullets (always include): {self.core_bullet_ids}"""

        try:
            # Run RLM retrieval
            result = rlm.completion(prompt=context, root_prompt=root_prompt)
            response = result.response

            # Track the raw RLM response
            self._selection_metadata["rlm_raw_response"] = response[:2000] if response else ""

            # Parse the response to extract bullet IDs
            selected_ids = self._parse_bullet_ids(response)

            # If parsing from response failed, try extracting from RLM log (REPL outputs)
            if not selected_ids and logger:
                if self.verbose:
                    print("[PlaybookRetriever] Response parsing failed, extracting from REPL outputs...")
                selected_ids = self._extract_ids_from_rlm_log(logger.log_file_path)
                self._selection_metadata["extracted_from_repl"] = True

            self._selection_metadata["rlm_selected_ids"] = selected_ids.copy()
            self._selection_metadata["parse_success"] = len(selected_ids) > 0

            if self.verbose:
                print(f"[PlaybookRetriever] RLM selected {len(selected_ids)} bullets")
                print(f"[PlaybookRetriever] Execution time: {result.execution_time:.2f}s")

        except Exception as e:
            print(f"[PlaybookRetriever] RLM retrieval failed: {e}")
            print("[PlaybookRetriever] Falling back to core bullets + heuristic selection")
            selected_ids = self._fallback_selection(task_instruction, all_bullets)
            self._selection_metadata["used_fallback"] = True
            self._selection_metadata["rlm_selected_ids"] = selected_ids.copy()

        # Ensure core bullets are included
        for core_id in self.core_bullet_ids:
            if core_id not in selected_ids and core_id in bullet_lookup:
                selected_ids.append(core_id)

        # Get the actual bullet objects
        selected_bullets = []
        for bullet_id in selected_ids:
            if bullet_id in bullet_lookup:
                selected_bullets.append(bullet_lookup[bullet_id])

        # Remove duplicates while preserving order
        seen = set()
        unique_bullets = []
        for bullet in selected_bullets:
            if bullet['id'] not in seen:
                seen.add(bullet['id'])
                unique_bullets.append(bullet)

        # Update final selection metadata
        self._selection_metadata["final_selected_ids"] = [b['id'] for b in unique_bullets]
        self._selection_metadata["final_bullet_count"] = len(unique_bullets)

        if self.verbose:
            print(f"[PlaybookRetriever] Final selection: {len(unique_bullets)} bullets")

        # Format back to playbook text
        final_playbook = format_bullets_as_playbook(unique_bullets, playbook_text)

        # Write selection log to file if log_dir is specified
        if self.log_dir:
            self._write_selection_log(unique_bullets, final_playbook)

        return final_playbook

    def _parse_bullet_ids(self, response: str) -> list[str]:
        """Parse bullet IDs from RLM response.

        Handles various formats:
        - JSON list: ["id1", "id2", ...]
        - Python list repr: ['id1', 'id2', ...]
        - Plain list: id1, id2, ...
        - Newline separated: id1\nid2\n...
        """
        # Try JSON parsing first (handles both JSON and Python list repr)
        try:
            # Find any array pattern in response (handles both " and ' quotes)
            # Look for patterns like ['shr-00001', 'shr-00005'] or ["shr-00001", "shr-00005"]
            array_match = re.search(r'\[([^\]]+)\]', response)
            if array_match:
                array_content = array_match.group(1)
                # Replace single quotes with double quotes for JSON parsing
                json_str = '[' + array_content.replace("'", '"') + ']'
                ids = json.loads(json_str)
                if isinstance(ids, list) and len(ids) > 0:
                    # Filter to only valid bullet ID patterns
                    valid_ids = [str(id_).strip() for id_ in ids
                                if id_ and re.match(r'^[a-z]{2,4}-\d{5}$', str(id_).strip(), re.IGNORECASE)]
                    if valid_ids:
                        return valid_ids
        except (json.JSONDecodeError, ValueError):
            pass

        # Try finding bullet ID patterns anywhere in the response
        id_pattern = r'\b([a-z]{2,4}-\d{5})\b'
        matches = re.findall(id_pattern, response, re.IGNORECASE)
        if matches:
            return list(dict.fromkeys(matches))  # Remove duplicates, preserve order

        # Fallback: empty list
        return []

    def _extract_ids_from_rlm_log(self, log_file_path: str) -> list[str]:
        """Extract selected_ids from RLM log file by parsing REPL outputs.

        When the RLM's final response doesn't contain valid IDs (e.g., model
        hallucination), we can recover the computed selected_ids from the
        code execution outputs stored in the log.

        Args:
            log_file_path: Path to the RLM log file (JSONL format).

        Returns:
            List of bullet IDs extracted from REPL outputs, or empty list if not found.
        """
        if not os.path.exists(log_file_path):
            return []

        selected_ids = []
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get('type') != 'iteration':
                        continue

                    # Check code_blocks for selected_ids in stdout or locals
                    code_blocks = entry.get('code_blocks', [])
                    for block in code_blocks:
                        result = block.get('result', {})

                        # Try to extract from stdout (e.g., JSON dump of selected_ids)
                        stdout = result.get('stdout', '')
                        if 'shr-' in stdout or 'api-' in stdout or 'code-' in stdout:
                            # Look for JSON array in stdout
                            ids = self._parse_bullet_ids(stdout)
                            if len(ids) > len(selected_ids):
                                selected_ids = ids

                        # Try to extract from locals
                        locals_dict = result.get('locals', {})
                        if 'selected_ids' in locals_dict:
                            local_ids = locals_dict['selected_ids']
                            if isinstance(local_ids, list) and len(local_ids) > len(selected_ids):
                                # Validate IDs
                                valid_ids = [
                                    str(id_).strip() for id_ in local_ids
                                    if id_ and re.match(r'^[a-z]{2,4}-\d{5}$', str(id_).strip(), re.IGNORECASE)
                                ]
                                if valid_ids:
                                    selected_ids = valid_ids

            if self.verbose and selected_ids:
                print(f"[PlaybookRetriever] Extracted {len(selected_ids)} IDs from REPL outputs")

        except Exception as e:
            if self.verbose:
                print(f"[PlaybookRetriever] Failed to extract from log: {e}")

        return selected_ids

    def _fallback_selection(
        self,
        task_instruction: str,
        all_bullets: list[dict],
    ) -> list[str]:
        """Fallback selection using simple keyword matching.

        Used when RLM retrieval fails.
        """
        task_lower = task_instruction.lower()
        scored_bullets = []

        # Simple keyword scoring
        keywords = {
            'spotify': ['spotify', 'music', 'playlist', 'song', 'album'],
            'phone': ['phone', 'text', 'message', 'contact', 'call', 'sms'],
            'gmail': ['email', 'gmail', 'mail', 'inbox', 'message'],
            'amazon': ['amazon', 'order', 'product', 'buy', 'purchase', 'cart'],
            'todoist': ['todo', 'task', 'todoist', 'reminder'],
            'splitwise': ['expense', 'split', 'debt', 'money', 'payment'],
            'file': ['file', 'document', 'folder', 'directory'],
            'simple_note': ['note', 'simple_note'],
            'venmo': ['venmo', 'pay', 'transfer'],
            'pagination': ['page', 'pagination', 'all', 'list', 'every'],
            'time': ['time', 'date', 'yesterday', 'today', 'week', 'month', 'year'],
            'api': ['api', 'endpoint', 'request'],
        }

        # Find which keyword categories are relevant to the task
        relevant_categories = set()
        for category, kws in keywords.items():
            for kw in kws:
                if kw in task_lower:
                    relevant_categories.add(category)

        # Score each bullet
        for bullet in all_bullets:
            content_lower = bullet['content'].lower()
            score = 0

            # Check for category keywords in bullet
            for category in relevant_categories:
                for kw in keywords.get(category, []):
                    if kw in content_lower:
                        score += 2

            # Bonus for helpful count
            score += bullet.get('helpful', 0) * 0.5

            # Penalty for harmful count
            score -= bullet.get('harmful', 0) * 0.5

            # Bonus for being in STRATEGIES section
            if 'strateg' in bullet.get('section', '').lower():
                score += 1

            scored_bullets.append((bullet['id'], score))

        # Sort by score and take top max_bullets
        scored_bullets.sort(key=lambda x: x[1], reverse=True)
        return [b[0] for b in scored_bullets[:self.max_bullets]]

    def _write_selection_log(
        self,
        selected_bullets: list[dict],
        final_playbook: str,
    ) -> None:
        """Write the selection log to a file in the log directory.

        Creates a file named 'selected_playbook.json' containing:
        - Selection metadata (task, counts, success/failure)
        - List of selected bullet IDs with their content
        - The final formatted playbook text
        """
        from datetime import datetime

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "metadata": self._selection_metadata,
            "selected_bullets": [
                {
                    "id": b["id"],
                    "section": b.get("section", "general"),
                    "content": b["content"],
                }
                for b in selected_bullets
            ],
            "final_playbook": final_playbook,
        }

        log_file_path = os.path.join(self.log_dir, "selected_playbook.json")
        try:
            with open(log_file_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            if self.verbose:
                print(f"[PlaybookRetriever] Selection log written to: {log_file_path}")
        except Exception as e:
            print(f"[PlaybookRetriever] Warning: Failed to write selection log: {e}")
