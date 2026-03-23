# Repository Guidelines

- Repo: https://github.com/picsalex/yolo-scout
- In chat replies, file references must be repo-root relative only (example: `yolo_scout/utils/plugins.py:80`); never absolute paths or `~/...`.

## Core Principles

Less is more. The simplest solution is the best solution.

The action hierarchy for every change: Delete > Replace > Add. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

    Minimal: The simplest solution that works. Don't over-engineer, don't over-abstract, don't add code "just in case." Three similar lines beat a premature abstraction. No error handling for impossible states. No feature flags or compatibility shims — just change the code.
    Solve at the source: Don't hack fixes — truly solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, never add workarounds, never add synchronization code for state that shouldn't be duplicated in the first place.
    Delete ruthlessly: When you replace code, delete what it replaced. Dead code accumulates fast — every change must leave the codebase cleaner. Remove unused imports, functions, types, files. Delete commented-out code. Git preserves history. Run bun run knip to verify.
    Replace > Add: Modify existing code over adding new. Edit existing files, extend existing components with a prop, parameterize existing utilities. If you're about to create a new file, stop — can it go in an existing file instead?
    Check existing: This is a monorepo — search ALL apps and @ultralytics/ui before creating anything new. If a feature exists in one app, reuse or adapt it. Shared code goes in @ultralytics/ui, then delete the original.
    Production ready: All changes must be thoroughly debugged and production ready.

When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"
