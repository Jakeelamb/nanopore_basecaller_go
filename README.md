# Go nanopore basecaller port

A separate Go port of caller computations from Dorado 2.1.2, pinned to
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.

The port is in progress. It does not yet accept POD5 and produce basecalls.
Inference, CRF decoding, input loading, splitting, scheduling, and final caller
output remain open. The original experiment repository is the stock control.

Changes use feature branches and pull requests targeting `main`. See
[AGENTS.md](AGENTS.md) for the experiment and evidence rules.

Derived Dorado code retains the upstream research-use license in
[LICENCE.dorado.txt](LICENCE.dorado.txt). See [NOTICE](NOTICE).
