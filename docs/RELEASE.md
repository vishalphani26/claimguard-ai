# Public package preparation

Recovered the backend from the original notebook's code-generation cell. Repaired malformed node indentation, added a missing regulatory grader prompt, corrected graph routing, and rebuilt the Streamlit form to match the backend's structured auto-claim schema. Added stricter relevance and grounding gates, separate index identifiers, environment-based model configuration, and synthetic policy files.

The original notebook, execution outputs, private documents, personal attribution details, original sample data, logs, vector stores and Git history were not copied. No actual API keys are needed in this package. The original files remain unchanged. A pattern-based scan is not a guarantee that every possible secret has been detected. Review before public release and obtain permission from contributors. The existing repository license is retained.

Validation: four static tests passed; Python sources compile. Live dependencies, provider calls, Streamlit launch, Docker and CI were not tested in the packaging environment. The original notebook is omitted rather than republished to minimize accidental disclosure.
