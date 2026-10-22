# Boundary reminder handbook

The work model can call `get_boundary_hints` to read selected cards from a bundled,
English-language handbook. The fixed tool description lists all twelve keys with
short descriptions; card bodies enter model history only when requested. The tool
does not search project files, use the network, execute commands or call another
model. Multiple distinct keys can be requested in one call.

| Key | Scope |
| --- | --- |
| `general` | Allowed inputs, assumptions, counterexamples and original goals |
| `math` | Domains, singularities, limits and probabilities |
| `numeric` | Precision, overflow, rounding, money and units |
| `collections` | Empty collections, singleton cases, indexing and ordering |
| `text` | Unicode, whitespace, encoding, escaping and parsing |
| `state` | Initialization, lifecycle, cleanup and recovery |
| `time` | Time zones, calendar boundaries and clock behavior |
| `permissions` | Identity, authorization, scope changes and thresholds |
| `concurrency` | Races, transactions, ordering and idempotence |
| `external_io` | Partial operations, dependency failures and retries |
| `generation` | Actual artifacts, conversion and end-to-end consumption |
| `interaction` | Empty, loading, error, cancellation and repeated-action flows |

Each card has five reminders. They ask what to check, not what the answer must be.
Requirements and independent public evidence still determine expected behavior.
No task IDs, framework-specific commands, reference answers or hidden grading
materials are part of the handbook.

## Acceptance integration

Before the first generated acceptance draft is accepted for original-source trial,
the host requires a saved lookup containing `general`. A lookup of another card
does not satisfy this prerequisite. The model selects other relevant categories;
there is no keyword classifier. Missing prerequisites reject the draft before its
trial or proposed edits run. Fixed user verification commands keep their existing
path and do not require a generated-acceptance lookup.

Applicable reminders belong in the existing `test_code` and `execution_checks`.
The existing freeze `reason` can briefly describe coverage and material gaps.
There is no new coverage form or independent reviewer. The host verifies lookup
receipts and real execution; it cannot prove that the model selected every useful
card, understood it or wrote complete tests. A read receipt is not a correctness
certificate.

Requirements or acceptance changes call for reconsidering which cards apply.
Reuse text still visible in context; cards can be reread after compaction. Reading
a card never authorizes weakening frozen checks or bypassing evidence rules.

## Records, recovery and cost

Lookups save the original model response, selected keys, catalog hash and returned
cards in task-private records, then append the paired tool result. A saved complete
response can finish its local lookup after interruption. A saved lookup record is
reused instead of issuing another model request. Changed or missing receipts stop
acceptance; persistence failures stop the task. Unknown model responses retain the
existing no-replay rules.

The catalog, schema and handler are part of the source identity and portable package.
Older tasks require their matching frozen source; this feature does not migrate
old checkpoints. Tool definitions and key order remain fixed across task phases.

The local tool has no provider charge, but model requests selecting keys and reading
returned cards use the normal task budget and usage ledger. Read cards remain in
history until ordinary compaction. On-demand loading avoids unrequested card bodies;
it does not establish lower total cost or improved task success without measurement.
