# Settings and binding audit, September 2026

The review covered the pending SwiftUI/settings diff, every custom binding
constructor in the Mac app, profile persistence transitions, and selected
Metal inference boundaries. It is not a proof that the entire engine is free
of defects.

## Findings and changes

- Non-selected model profiles previously came from disk without observable
  state. Editing a profile could leave sibling controls and memory estimates
  stale. Profiles now live in the observable settings store and update before
  the coalesced disk write.
- General settings, updater preferences, and server controls had redundant
  binding adapters. Direct bindings now reach setters that own persistence
  or validation. The profile dictionary has an internal setter so views
  cannot bypass profile validation and dirty tracking.
- Pending profile edits are saved before changing the model directory or
  deleting the selected conversation. Restoring a conversation during startup
  does not introduce a settings write.
- Failed writes for non-selected profiles retain their dirty state and can
  be retried by a later save or quit flush. Profiles sharing a settings file
  are merged without overwriting each other's edits. Auto retains the manual
  context, cache, and prefill values underneath its effective plan.
- Two GPU argmax helpers could return an invalid token ID when all logits
  were non-finite. They now use the fused heads' token-0 fallback; the FP16
  helper also ignores non-finite candidates consistently.
- The server installs its non-Sendable idle handler synchronously on its
  channel event loop, removing the concurrency warning without moving the
  handler across threads.

## Remaining custom bindings

`Binding` is a property-wrapper value type, not a macro. Constructing a binding
with getter and setter closures does not inherently break Observation.
Observation depends on the state read by the view and mutated by the setter.

| Binding | Reason retained |
| --- | --- |
| Chat and server model selection | Validate switches against active work and model capabilities. |
| Rename presentation | Derive presentation from an optional target and clear it on dismissal. |
| Prompt editor focus | Bridge `FocusState.Binding` to the AppKit editor's plain binding. |
| Settings profile selection | Follow the current model until an explicit profile is chosen. |
| Generic profile field | Validate the edited value profile and schedule persistence. |
| Automatic memory | Restore saved manual values when Auto is disabled. |
| Expert cache slots | Disable prefill when slots fall below its supported floor. |
| Accent color and hex text | Convert representations and retain partial input until valid. |

No manual Sync button was found. Reload Model applies changes to an already
loaded runtime; update and health checks perform explicit external operations.
These are meaningful actions rather than substitutes for observing state.

The app already uses protocols at inference, lifecycle, installer, and server
backend boundaries. A blanket protocol conversion would not fix the observed
settings bug. The large AppModel remains a maintainability concern that calls
for focused extraction around concrete responsibilities.

## Verification scope

Regression coverage includes observation invalidation, non-selected edits
without persistence, derived memory estimates, shared-file merging, model
switches, Auto/manual restoration, invalid profile rejection, directory and
conversation transitions, and retry after a failed write. GPU tests cover
finite ties, mixed NaN/infinities, all-invalid fallback, and scalar/batched
projection equivalence. `Scripts/test.sh` is the canonical serial gate.
The final local run passed 1,503 tests in 241 suites.

Speculative decoding remains opt-in. Gemma and GPT-OSS have transaction
equivalence fixtures; the generic MiniMax path does not yet have equivalent
model-specific rollback coverage. This release makes no new live-model
accuracy or performance claim.
