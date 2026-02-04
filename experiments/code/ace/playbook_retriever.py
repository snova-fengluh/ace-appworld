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


# Custom system prompt for playbook retrieval
RETRIEVER_SYSTEM_PROMPT = """You are a playbook retrieval assistant. Your task is to analyze a task instruction and select the most relevant guideline bullets.

The context contains:
- `task_instruction`: The task to be performed
- `bullets`: A list of guideline bullets, each with 'id', 'section', and 'content' (content may be truncated)
- `app_descriptions`: Available apps for this task
- `max_bullets`: Maximum number of bullets to select
- `core_bullet_ids`: Bullet IDs that must always be included

Your job is to:
1. Look at the task instruction and available apps
2. Scan through the bullets to find relevant ones
3. Select the most relevant bullet IDs for the task
4. Return the IDs as a Python list

Strategy - be efficient:
1. First check what apps are mentioned in the task (spotify, phone, venmo, etc.)
2. Look for bullets mentioning those apps or related operations
3. Include bullets about general strategies (pagination, API lookup, etc.)
4. Select up to max_bullets most relevant IDs

IMPORTANT - How to return your final answer:
- Create a Python list variable with the selected bullet IDs
- Then OUTSIDE the code block, write: FINAL_VAR(variable_name)

Example:
```repl
# Quick scan for relevant bullets
task = context['task_instruction'].lower()
relevant = []
for b in context['bullets']:
    if 'spotify' in task and 'spotify' in b['content'].lower():
        relevant.append(b['id'])
    # Add more matching logic...
selected = context['core_bullet_ids'] + relevant[:context['max_bullets']]
print(selected)
```
FINAL_VAR(selected)

DO NOT call FINAL() or FINAL_VAR() inside the code block.
"""


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
                    playbook_retriever_{timestamp}_{uuid}.jsonl

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
        # Parse all bullets from playbook
        all_bullets = parse_playbook_bullets(playbook_text)

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
            custom_system_prompt=RETRIEVER_SYSTEM_PROMPT,
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

            # Parse the response to extract bullet IDs
            selected_ids = self._parse_bullet_ids(response)

            if self.verbose:
                print(f"[PlaybookRetriever] RLM selected {len(selected_ids)} bullets")
                print(f"[PlaybookRetriever] Execution time: {result.execution_time:.2f}s")

        except Exception as e:
            print(f"[PlaybookRetriever] RLM retrieval failed: {e}")
            print("[PlaybookRetriever] Falling back to core bullets + heuristic selection")
            selected_ids = self._fallback_selection(task_instruction, all_bullets)

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

        if self.verbose:
            print(f"[PlaybookRetriever] Final selection: {len(unique_bullets)} bullets")

        # Format back to playbook text
        return format_bullets_as_playbook(unique_bullets, playbook_text)

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
