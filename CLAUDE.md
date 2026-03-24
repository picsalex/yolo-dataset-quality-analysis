# Repository Guidelines - yolo-scout

- **Repo:** https://github.com/picsalex/yolo-scout
- **References:** Use repo-root relative paths only (e.g., `yolo_scout/utils/plugins.py:80`). Never use absolute paths or `~/`.

## Core Principles

**Minimalism is the priority.** The action hierarchy for every change: **Delete > Replace > Add.**

1. **Delete:** The best code is deleted code. Remove unused imports, dead functions, and commented-out blocks. Do not leave "just in case" code; Git preserves history.
2. **Replace:** Modify or parameterize existing code rather than adding new layers. If a feature exists, adapt it.
3. **Add:** New code is a last resort. Do not over-engineer or add abstractions for hypothetical future states.

## Execution Standards

- **Solve at the Source:** Fix root causes, not symptoms. Never patch over a broken abstraction; fix or remove the abstraction itself.
- **No Redundancy:** Search the codebase for existing utilities before creating new ones. Reuse and consolidate logic to prevent duplication.
- **Production Ready:** All changes must be thoroughly debugged, typed, and functional.
- **Cleanliness:** Run linting and formatting tools `ruff`, `pytest` to ensure the codebase remains clean and stable.

## Communication

- Be concise.
- Stick to the facts.
- When fixing bugs, prioritize the "What can I delete?" mindset.
