---
name: handover
description: Print a short, copy-pasteable handoff — recent progress and what's next, nothing else — for a new Claude session or a different machine to pick up cold. Use when the user invokes /handover, or says things like "hand this off", "I'm starting a new session", "running low on context", "prep this for the cluster/VM", or "what do I tell the next session".
---

# Handover

A short note for someone with none of this conversation — a fresh
session, or a different machine. Just enough to pick up, nothing more.

## When invoked

Print one fenced block:

- **Title** — one line that says this IS a handover from a prior
  session, plus the goal, so whoever reads it (a new agent, cold)
  immediately knows what's happening: "Handover from a previous
  session — continuing: <the goal, like an experiment title>".
- **Recent** — 2-4 bullets, what just happened. Point at a commit hash
  or `experiments_log.md` entry instead of re-explaining it.
- **Next** — 1-3 bullets, the concrete next step(s), in order.

If something is actually running right now, add its literal command as
a bullet under Next (not its own section).

If the work actually splits across machines — some done or doable
locally, some needing a GPU VM/cluster — say which is which, inline in
the bullet it applies to (e.g. "on Delta: ..."). Skip this entirely when
everything is local; don't add a location to every bullet just because
one needed it.

That's the whole thing. No headers beyond Recent/Next, no restating the
project, no caveats section. If it's longer than ~8 bullets total,
cut it down.

## What this is not

- Not `/todos`: that's for you, now, in this chat.
- Not `experiments_log.md`: that's permanent results history. This never
  touches disk unless the user says to save it.
