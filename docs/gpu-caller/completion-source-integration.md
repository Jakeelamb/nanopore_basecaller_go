# Caller source integration checkpoint

This branch combines the reviewed SUP consumer prerequisites, scheduler CLI,
fixed partial-publication accounting, immutable packed host plan, watcher exit
correction, and packed HAC source preparer/generator. It targets `main`; none of
the prerequisite pull requests has been merged.

Validation on 2026-10-29:

- `CGO_ENABLED=1 go test -p 2 -json ./...`: 900 root tests and 2,087 subcases
  pass; 93 opt-ins are skipped.
- The maintained `verification/verify.sh` command passes its CPU-package and
  native-algorithm scope.
- Each new source unit has two completed independent same-model source reviews;
  unchanged reviews are reused. The combined branch has the integration checks
  above, rather than an additional whole-branch review panel.
- Retained genuine CPU auxiliary evidence and corruption controls match the
  packed plan's first-case host arrays. The native producer was not repeated.
- The built Go0 source CLI generates five pinned source copies. Independent
  reversal recovers all original bytes; changed-source and existing-output
  controls reject. Its native observer remains uncompiled.

The approved GPU registry remains empty. The genuine CPU13 receipt qualifies
only its declared operations and exact backend tuple. SUP compilation and
selected-artifact validation do not establish GPU graph acceptance. The SUP
metadata author is being simplified; its failed transitive-inventory attempts
are retained and are not integrated here.

Remaining work includes the genuine SUP admission and numerical run, fixed HAC
graph admission, packed HAC/SUP native producers and kernel bindings, segmented
decoding, collision controls, partial GPU publication, complete caller outputs,
development/validation cohorts, variant retention, biological and timing checks.
HG005 is excluded from selection. Stock source, controls and the original dirty
Go checkout remain unchanged. Raw signals, weights, generated payloads, receipts
and results remain outside Git.
