# Design notes

## Why agentic retrieval?
A vector search always finds nearest neighbors, even when none resolve a claim. The relevance gate and bounded rewrite loop are intended to catch weak evidence before reasoning. A second evidence gate checks grounding and citation membership after generation. Neither LLM grader is a guarantee of correctness.

## Tradeoffs
- Chroma and MiniLM provide a lightweight local retrieval prototype; hosted inference handles language reasoning.
- Separate grader and decision model configurations allow future latency and quality experiments; no savings have been measured here.
- Typed state and structured claim input make decisions easier to reproduce and inspect than free-form outputs.
- Escalation over guessing prioritizes review when coverage evidence is incomplete. Regulatory context is advisory, not a substitute for a policy clause.

## Next steps
Add deterministic policy rules, source allowlists with independent validation, adversarial evaluation, integration tests with mocked providers, a labeled evaluation corpus, access controls and protected audit storage. Benchmark before making performance claims.
