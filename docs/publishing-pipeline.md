# Publishing Pipeline

## Commands

- `npm run post:agent -- plan --topic "..."` previews the file path and route.
- `npm run post:agent -- draft --topic "..."` writes a new source MDX post into `content/posts/agentic/`.
- `npm run post:agent -- publish --topic "..." --target-branch main` writes the MDX post, runs validation, commits it, and pushes it.
- `npm run refresh:popular` refreshes `_data/analytics.json` and `_data/analytics_month.json`.
- `npm run typecheck:publishing` typechecks only the publishing/workflow layer.

## Validation Contract

- `publish` uses `npm run validate:publish`.
- `validate:publish` is expected to call `npm run typecheck:publishing`, then the shared content generation and validation scripts, then the frontend validation script.
- The pipeline does not duplicate content validation or route parity logic.

## Environment And Secrets

- `OPENAI_API_KEY`: enables model-backed drafting.
- `OPENAI_API_BASE_URL`: optional base URL for an OpenAI-compatible endpoint. Defaults to `https://api.openai.com/v1`.
- `POST_AGENT_MODEL`: model identifier for drafting.
- `GOOGLE_ANALYTICS_CREDENTIALS_JSON`: service account JSON for GA4 refresh.
- `GA_PROPERTY_ID`: optional GA4 property id. Defaults to `397192433`.

## Guardrails

- `publish` refuses to run when the worktree has unrelated changes outside the post and generated build paths.
- `publish` commits only the authored MDX file under `content/posts/`.
- If pushing directly to the target branch fails, the script pushes a fallback branch instead of discarding work.
- Draft generation falls back to a deterministic scaffold when model credentials are unavailable.
