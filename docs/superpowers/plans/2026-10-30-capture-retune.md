# Shared capture retune repair

**Goal:** Keep the receiver usable through an intentional frequency change without relabeling old IQ or joining samples from different tunings.

**Architecture:** Retain one shared capture service, budget, writer workspace and header worker. Give bands and views immutable input-center metadata. An explicit retune seals the old bands, saves the contiguous IQ already retained for unfinished full views as `shared_partial`, preserves complete/writer-owned views, and starts a fresh reader generation. Uncommanded discontinuities still fail visibly. The detector resets its sliding history and rejects delayed results from the old generation.

**Evidence:** The previous shared live lifecycle run terminated when `advance` saw the generation increment from the existing control handler. The handler changed mutable recorder frequency metadata while queued saves still used it. The legacy path stayed alive through the same transitions.

**Constraints:** No hardware or CPU benchmarks concurrently with other tests. Preserve original and candidate baselines. No Pi changes or pushes. Retunes cannot invent missing tail samples or represent an interrupted view as complete. All retained arrays remain inside the same aggregate byte budget.

## Review focus

- Old complete and writer-owned IQ must retain its original frequency/phase and remain readable.
- Partial saves contain only emitted samples plus actual FIR support, with interruption counters and explicit file role.
- A crop or metadata snapshot in flight during the reset cannot kill or rewind the new generation.
- Delayed header results cannot create early views in a different generation; their leases remain owned until worker release.
- Old detector windows cannot be registered against new frequency metadata.
- Multiple retunes share one budget/service and release obsolete spans.

## Tasks

- [x] Add deterministic real-ring tests for idle, collecting, ready/writer-owned, and concurrent crop retunes; verify RED before implementation.
- [x] Add immutable center metadata and explicit generation rollover, retaining received prefixes without fabricated tails.
- [x] Test header cancellation, stale registration, recorder output, and main sliding-state reset; implement the matching integration.
- [ ] Run focused and full tests, then the retained native replay and live lifecycle suite with fresh source provenance.
- [ ] Review the complete change and retain evidence before committing.

Software validation: initial RED 7 core failures, then 4 integration failures; core GREEN 46, integration GREEN 89. Main-loop generation tests RED 8 failed / 3 passed, then GREEN 11; related regression set 189 passed. Partial decoder-role RED 4 failed / 20 passed, then GREEN 24 including actual saved IQ, concurrent prefix writing, and per-reference endpoints. Integrated full suite: 772 passed in 19.68 seconds. Independent software lifecycle review found no blocker. Native/live validation is still pending; no precise hardware retune boundary is claimed because streamer and detector control polling remain independent.
