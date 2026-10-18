# Migration notes

This is a runtime-aware client update bundle, not a commit to the GitHub repository.

1. Merge `src/arc_v2_inference_client/` into the SDK repository.
2. Merge the package metadata and wheel package list from `pyproject.toml` into the existing file; preserve any repository-specific dependencies, scripts, and tooling settings. Keep `src/arc_inference` in the build during migration.
3. Replace the README with this bundle's README or port its sections into the existing document.
4. Run `python -m pytest -q`, then run the live integration checks against the running inference runtime.
5. Migrate imports to `arc_v2_inference_client`; leave `arc_inference` available until downstream consumers are migrated.
6. Rename the GitHub repository only after downstream Git URLs, docs, and local `uv` sources are ready.

## Runtime paths represented

This bundle targets the routes documented by the inference-runtime upgrade and exercised by its integration test:

- `/v1/model`, `/v1/models`
- `/v1/tokenize`
- `/v1/chat/context`
- `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- `/v1/conversations` and `/v1/conversations/{id}/chat/completions`

Conversation completions send **new messages only**. The server persists the assistant/tool messages and is responsible for pruning old turns to fit the active context window.

## Compatibility note

The current GitHub repository's public README includes higher-level agent tracing, tool-call retry behavior, and structured output. This update bundle provides a simpler runtime-native agent baseline and stream events. Before replacing the original implementation wholesale, merge these modules into the repository and port any additional tracing/retry behavior required by current consumers.
