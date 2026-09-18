---
name: todos
description: Show a quick snapshot of what Claude is doing right now and what's next — a status check, not a report. Use when the user invokes /todos, or asks things like "what are you doing right now", "what's the status", "where are we", "what's left", or "give me a todo list".
---

# Todos

A glance, not a recap.

## When invoked

First line: the session's high-level goal, one plain sentence — e.g.
"Reproducing the entire results and figures using NSE and KGE metric for
hyperparam selection." Not a step, the overarching thing the steps serve.

Then scan the session — active task, background jobs/wakeups/subagents,
what's already been said — and list as markdown bullets, ordered done →
in progress → next, letting the symbol carry the status instead of headers:

- `- [x]` done (only if still relevant to what follows — skip stale history)
- `- [~]` in progress right now
- `- [ ]` next
- `- [!]` blocked on a decision only the user can make

One line each, no sub-bullets, no preamble beyond the title line, fits on
one screen.

## What this is not

- Not a plan: report decided steps, don't propose new ones.
- Not work: read state, don't change it.
- Not `/echo`: a status check, not a comprehension check — point there instead if that's the actual ask.
