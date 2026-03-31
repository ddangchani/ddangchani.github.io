# Integration Contracts

## Shared Types

- `PostMeta`
- `SearchEntry`
- `PopularPostEntry`
- `PostJobInput`
- `PostJobResult`

## Routing

- Preserve legacy public URLs exactly.
- Derive canonical routes from `routeSegments`.
- Do not regenerate routes from rewritten slugs or titles.
- Preserve Korean, spaces, parentheses, and encoded path behavior.

## Static Artifacts

- Search data must be generated as a stable build artifact.
- Popular-post data must be generated as a stable build artifact.
- Route manifest and route parity output must be available to validation scripts.
- Assets under `/assets/...` must remain addressable in the first React release.

## Validation Entry Points

- Frontend validation should remain callable from npm scripts.
- Content validation and route parity checks should remain callable from npm scripts.
- Publish validation must call shared npm entrypoints rather than duplicating repo logic.

## Publishing Guardrails

- `publish` pushes only on explicit publish intent.
- `publish` runs validation before git actions.
- `publish` commits only generated post-related files.
- If direct push is blocked, fall back to branch or PR flow instead of forcing failure.
