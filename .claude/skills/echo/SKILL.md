---
name: echo
description: >-
  Restate the user's request as a cleaned-up draft spec and pause before doing any actual work — before planning, before auto-mode execution, before editing files, before anything. The restatement is the real point, not just a comprehension check: it's meant to become the working prompt. Whatever the user sends back (confirmation, a correction, or an edited rewrite of the restatement itself) replaces their original raw prompt as the instruction to follow. Use whenever the user invokes /echo, or says things like "echo that back", "say that back to me", "make sure you understood before you start", "confirm you got this right", or "repeat my ask before you go". Also use it when the user asks to make this a standing habit ("turn on echo mode", "always confirm before starting") — keep restating-and-pausing on every substantive request until they say to stop. Skip it for trivial one-line questions with nothing to misunderstand.
---

# Echo

The user's original message is a first draft, your restatement is a second
draft, and whatever comes back after that — confirmation, correction, or
their own rewrite — is the final draft you actually work from, not the
original prompt.

## When triggered

1. Stop before acting — no planning, plan mode, state-changing tools, or
   auto-mode execution until this step is done.
2. Restate the request as a clean, standalone spec: goal, scope (what's in
   and explicitly out), constraints, and intended mode (plan/auto/direct) if
   relevant. A few sentences or bullets — don't pad it out if the ask was
   already a short, unambiguous sentence.
3. End with an explicit pause — "Is that right, or anything to tighten?" —
   and actually wait. Don't assume yes and continue in the same turn.
4. Work from whatever comes back, not the original prompt:
   - Confirmed as-is → your restatement is the spec.
   - Corrected in chat → the merged version is the spec.
   - Edited directly (rewritten, or edited in the IDE) → their edit is the
     spec verbatim, even where it overrides their own original wording.
5. Proceed using that finalized spec.

## Standing mode

"Turn on echo mode" (or similar) → repeat this for every substantive request
until they say to stop ("turn off echo mode"). Skip trivial one-liners, same
judgment call as when to ask a clarifying question at all. To persist past
this session, save it as a feedback memory rather than relying on
in-conversation state.
