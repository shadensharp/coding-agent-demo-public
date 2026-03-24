"""Centralized prompts for the coding agent."""

SYSTEM_PROMPT = """You are a pragmatic coding agent.
Work only inside the given Python repository.
Stay grounded in the provided files and constraints.
Use only the file names, functions, fields, and commands that appear in the supplied context.
If the context is insufficient, say which listed file should be inspected next instead of inventing details.
Respond concisely and do not invent files, commands, frameworks, APIs, or requirements.
When asked for structured output, return a single raw JSON object with no markdown fences.
"""

CLARIFY_PROMPT = """Turn the user request into a short implementation target.
Mention only files and behaviors supported by the supplied snippets.
Use the exact validation command that is provided.
Prefer the retrieved files and file summaries when deciding what is most relevant.
Return a raw JSON object with this schema:
{
  "implementation_target": "short string",
  "relevant_files": ["path.py", "tests/test_path.py"],
  "validation_command": "exact command string"
}
"""

PLAN_PROMPT = """Generate a short ordered coding plan.
Keep it concrete, bounded to the current repository, and focused on execution.
Use the exact validation command that is provided.
Do not mention frameworks, endpoints, or tools not present in the snippets.
Prefer the retrieved files and file summaries when deciding what to inspect or edit first.
Return a raw JSON object with this schema:
{
  "steps": ["step 1", "step 2", "step 3", "step 4"],
  "target_files": ["path.py", "tests/test_path.py"],
  "validation_command": "exact command string"
}
"""

PROPOSAL_PROMPT = """Generate a proposal-only edit candidate.
This proposal is advisory and will not be applied directly.
Stay strictly inside the allowed edit scope and mention only listed files, functions, and commands.
Use the exact validation command that is provided.
Do not invent extra files, endpoints, frameworks, or commands.
Return a raw JSON object with this schema:
{
  "summary": "one short sentence",
  "edits": [
    {
      "path": "path.py",
      "change_type": "update",
      "target_symbols": ["function_name", "TestCase.test_name"],
      "intent": "what should change in this file"
    }
  ],
  "validation_command": "exact command string"
}
"""


def _format_bullets(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)



def _format_file_snippets(file_snippets: list[tuple[str, str]]) -> str:
    if not file_snippets:
        return "- none"

    sections: list[str] = []
    for path, snippet in file_snippets:
        sections.append(f"- file: {path}\n```python\n{snippet}\n```")
    return "\n\n".join(sections)



def _format_retrieved_files(retrieved_files: list[tuple[str, str]]) -> str:
    if not retrieved_files:
        return "- none"
    return "\n".join(f"- {path}: {reason_text}" for path, reason_text in retrieved_files)



def _format_file_summaries(file_summaries: list[tuple[str, str]]) -> str:
    if not file_summaries:
        return "- none"
    return "\n".join(f"- {path}: {summary_text}" for path, summary_text in file_summaries)



def build_clarify_prompt(
    task_text: str,
    repo_name: str,
    constraints: list[str],
    visible_files: list[str],
    retrieved_files: list[tuple[str, str]],
    file_summaries: list[tuple[str, str]],
    file_snippets: list[tuple[str, str]],
    validation_command: str,
) -> str:
    return (
        f"Task:\n{task_text}\n\n"
        f"Repository:\n- name: {repo_name}\n"
        f"- visible files:\n{_format_bullets(visible_files)}\n\n"
        f"Retrieved files:\n{_format_retrieved_files(retrieved_files)}\n\n"
        f"Retrieved file summaries:\n{_format_file_summaries(file_summaries)}\n\n"
        f"Validation command:\n- {validation_command}\n\n"
        f"Key file snippets:\n{_format_file_snippets(file_snippets)}\n\n"
        f"Constraints:\n{_format_bullets(constraints)}\n\n"
        "Return only the raw JSON object for the clarify artifact."
    )



def build_plan_prompt(
    task_text: str,
    clarify_summary: str,
    constraints: list[str],
    visible_files: list[str],
    retrieved_files: list[tuple[str, str]],
    file_summaries: list[tuple[str, str]],
    file_snippets: list[tuple[str, str]],
    validation_command: str,
) -> str:
    return (
        f"Task:\n{task_text}\n\n"
        f"Clarified target:\n{clarify_summary}\n\n"
        f"Visible files:\n{_format_bullets(visible_files)}\n\n"
        f"Retrieved files:\n{_format_retrieved_files(retrieved_files)}\n\n"
        f"Retrieved file summaries:\n{_format_file_summaries(file_summaries)}\n\n"
        f"Validation command:\n- {validation_command}\n\n"
        f"Key file snippets:\n{_format_file_snippets(file_snippets)}\n\n"
        f"Constraints:\n{_format_bullets(constraints)}\n\n"
        "Return only the raw JSON object for the plan artifact."
    )



def build_proposal_prompt(
    task_text: str,
    clarify_summary: str,
    plan_summary: str,
    read_summary: str,
    constraints: list[str],
    retrieved_files: list[tuple[str, str]],
    file_summaries: list[tuple[str, str]],
    file_snippets: list[tuple[str, str]],
    validation_command: str,
    allowed_edit_scope: list[str],
) -> str:
    return (
        f"Task:\n{task_text}\n\n"
        f"Clarified target:\n{clarify_summary}\n\n"
        f"Execution plan:\n{plan_summary}\n\n"
        f"Read evidence:\n{read_summary}\n\n"
        f"Retrieved files:\n{_format_retrieved_files(retrieved_files)}\n\n"
        f"Retrieved file summaries:\n{_format_file_summaries(file_summaries)}\n\n"
        f"Allowed edit scope:\n{_format_bullets(allowed_edit_scope)}\n\n"
        f"Validation command:\n- {validation_command}\n\n"
        f"Key file snippets:\n{_format_file_snippets(file_snippets)}\n\n"
        f"Constraints:\n{_format_bullets(constraints)}\n\n"
        "Return only the raw JSON object for the proposal candidate. Stay strictly inside the allowed edit scope above."
    )
