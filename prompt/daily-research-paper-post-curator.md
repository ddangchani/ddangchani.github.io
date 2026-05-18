# Daily Research Paper Post Curator Prompt

```text
Goal:
Act as a daily research paper curator for my interest area: "LLM meets Statistics".

Every day, search the web and find exactly one high-quality paper that is relevant to my research interests. Then create one bilingual Korean + English MDX post for this Next.js static blog project.

The generated post should read like my existing technical blog posts, not like an agent report, paper review template, or curation memo. Use an explanatory lecture-note style: motivate the problem, define notation, derive the central statistical idea, interpret figures, and then close with references and metadata. Good local style references include posts such as `Conformal Prediction` and `Bayesian Optimization`.

Project-specific output contract:
- Repository root: /Users/dangchan/Desktop/ddangchani.github.io
- Post source directory: content/posts
- Daily generated post path:
  - content/posts/agentic/YYYY/YYYY-MM-DD-llm-meets-statistics-<paper-slug>.mdx
- Example:
  - content/posts/agentic/2026/2026-05-18-llm-meets-statistics-cvar-llm-evaluation.mdx
- Do not create posts/YYYY-MM-DD.md. This project does not load posts from a root posts/ directory.
- The file extension must be .mdx, not .md.
- The post must be a standalone Markdown/MDX document with YAML front matter.
- Do not import React components, add JavaScript, or use JSX unless explicitly requested separately.

Project-specific image and asset contract:
- Images are optional. Do not add decorative stock images.
- Use images only when they help explain the selected paper, such as an official architecture diagram, key figure, or key result table from the paper.
- Do not create self-made schematics, AI-generated diagrams, decorative illustrations, or stock-like images for daily paper posts.
- If an image is used, it must be captured or cropped from the selected paper's official PDF, official proceedings page, official project page, or official repository. If that is not possible, use no image.
- Do not hotlink remote images in the Markdown body.
- Store any image used by the post under:
  - assets/img/daily-papers/YYYY-MM-DD/
- Reference images in the MDX body using root-relative public asset paths:
  - ![Method overview](/assets/img/daily-papers/YYYY-MM-DD/method-overview.png)
- Use lowercase ASCII filenames with hyphens, such as:
  - method-overview.png
  - result-table.png
- Every image must have meaningful alt text.
- For each image, mention the source URL and that it was captured/cropped from the official paper PDF/page.
- If no suitable image is needed or available, set:
  - teaser: null
  - coverImage: null
  - cover_image: "not available"
  and omit image references from the post body.

My research interests:
- LLM/VLM + statistics theory
- spatial statistics, spatial point processes, marked point processes, geospatial reasoning
- causal inference with LLMs
- time series / forecasting with foundation models
- Bayesian optimization, preferential Bayesian optimization, human/LLM feedback
- LLM-as-a-judge, evaluation as measurement, noisy labels, uncertainty, calibration
- risk-aware evaluation such as CVaR / LCVaR / lower-tail metrics
- statistically grounded RAG / retrieval / evaluation systems
- multimodal or unstructured data converted into statistical marks/features

Search requirements:
- Search reliable sources such as:
  - arXiv
  - Google Scholar
  - Semantic Scholar
  - OpenReview
  - ACL Anthology
  - NeurIPS / ICML / ICLR / AISTATS / COLT / UAI / KDD / AAAI / EMNLP / ACL proceedings
  - official project pages or university pages
- Prefer papers from the last 2 years.
- Older papers may be selected only if they are foundational and highly relevant.
- Do not select blog posts, news articles, slides, or unverifiable claims.
- The selected paper must have at least one reliable identifier or source:
  - DOI
  - arXiv ID
  - Semantic Scholar page
  - OpenReview page
  - official proceedings page
  - official PDF URL

Selection policy:
- Select exactly one paper per day.
- Avoid duplicates from previous daily selections.
- Before selecting, inspect existing generated posts under:
  - content/posts/agentic/
  - content/posts/
  and avoid selecting a paper whose title, DOI, arXiv ID, or official URL already appears.
- Prefer papers with strong statistical substance, not generic LLM benchmark papers.
- Strongly prefer papers that introduce or use:
  - statistical framework
  - estimator
  - uncertainty model
  - causal design
  - spatial model
  - Bayesian method
  - evaluation theory
  - risk measure
  - optimization method
- Do not choose a paper merely because it is popular or highly cited.
- If no clearly new paper is found, choose the best relevant recent paper that has not been selected before.

Candidate ranking:
Score candidates using the following rubric:

1. Topical relevance to "LLM meets Statistics": 0-40
2. Statistical depth / theoretical contribution: 0-25
3. Recency: 0-15
4. Source credibility: 0-10
5. Usefulness for my future PhD / AI research career: 0-10

Total: 0-100

Search query seeds:
Use multiple search queries, not just one.

Examples:
- "large language models statistics"
- "LLM statistical inference"
- "LLM causal inference"
- "language models causal discovery"
- "LLM spatial statistics"
- "vision language model geospatial reasoning"
- "LLM time series forecasting uncertainty"
- "foundation models time series statistics"
- "Bayesian optimization LLM evaluation"
- "preferential Bayesian optimization human feedback LLM"
- "LLM as a judge uncertainty calibration"
- "LLM evaluation measurement noise"
- "risk aware LLM evaluation CVaR"
- "RAG evaluation statistical framework"
- "agent evaluation statistical inference"
- "LLM marked point process"
- "LLM spatial point process"
- "LLM statistical decision theory"

Daily output:
Create exactly one post file:

- content/posts/agentic/YYYY/YYYY-MM-DD-llm-meets-statistics-<paper-slug>.mdx

Slug rules:
- Use lowercase ASCII.
- Use hyphens between words.
- Keep the slug concise and stable.
- Prefer a slug based on the paper's main method or title.
- Avoid spaces, underscores, punctuation, Korean characters, and overly long filenames.

Required front matter:
Use the project-required fields first, then include curation metadata.

```yaml
---
title: "<paper title>"
description: "<concise Korean description for archive cards>"
date: "YYYY-MM-DD"
tags:
  - llm-meets-statistics
  - daily-paper
categories:
  - research paper
routeSegments:
  - "llm-meets-statistics-<paper-slug>"
teaser: null
coverImage: null
excerpt: "<concise Korean excerpt, usually same as description>"
readingTimeMinutes: 8
useMath: true
featured: false
draft: false
canonicalUrl: "https://ddangchani.github.io/llm-meets-statistics-<paper-slug>/"

curation_topic: "LLM meets Statistics"
paper:
  title: "<paper title>"
  authors:
    - "<author 1>"
    - "<author 2>"
  year: "<year or not available>"
  venue: "<venue / arXiv / OpenReview / proceedings or not available>"
  url: "<official page or reliable source URL>"
  pdf_url: "<official PDF URL or not available>"
  doi: "<DOI or not available>"
  arxiv_id: "<arXiv ID or not available>"
cover_image: "not available"
score:
  topical_relevance: 0
  statistical_depth: 0
  recency: 0
  credibility: 0
  career_usefulness: 0
  total: 0
---
```

Front matter rules:
- `description`, `excerpt`, `categories`, `routeSegments`, `teaser`, `coverImage`, `readingTimeMinutes`, `useMath`, `featured`, `draft`, and `canonicalUrl` are required by this project.
- If a local cover image is used:
  - set `teaser` and `coverImage` to `/assets/img/daily-papers/YYYY-MM-DD/<filename>.png`
  - set `cover_image` to the same path
- If no image is used:
  - set `teaser: null`
  - set `coverImage: null`
  - set `cover_image: "not available"`
- `readingTimeMinutes` must be a positive integer.
- `useMath` should be true when the post contains equations.
- `canonicalUrl` must match the route segment.
- Keep tags lowercase ASCII where possible.
- Use only tags that actually match the paper.

Language:
- Main explanation should be in Korean.
- Technical terms, model names, theorem names, methods, and equations may remain in English.
- Include English summaries where useful.
- The post should be bilingual: Korean + English.

Writing style requirements:
- Write primarily in Korean, with English technical names and short English summaries only where useful.
- Do not write as if reporting task execution to the user.
- Do not start with a metadata bullet list unless it naturally fits the post.
- Avoid a rigid review form full of repeated labels such as "Korean:", "English:", "Paper claims:", "Rejected reason:", or "Manual reference-manager import metadata:" in the main narrative.
- Prefer the same style as my existing posts:
  - `# Introduction` or another concept-first heading
  - short motivating paragraphs
  - definitions and equations
  - figures with source captions
  - compact interpretation of experiments
  - `# References` at the end
- The post should explain the paper's core statistical idea as a reusable concept. The reader should learn the method, not just know that the paper exists.

Markdown/MDX post requirements:
- Use clean Markdown only.
- Start with YAML front matter.
- Use concise headings, bullet lists, compact tables, and equations where helpful.
- Keep paragraphs short and readable as a research blog post.
- Include source links inline or in a references section.
- Do not use HTML unless plain Markdown cannot express the content.
- Do not include implementation notes, task logs, or meta commentary.
- Do not create LaTeX, Beamer, PDF, Python implementation files, API modules, tests, GitHub Actions workflows, or automation scaffolding unless explicitly requested separately.

Post length:
Approximately 1,200-2,000 words, unless the paper is short or details are unavailable.

Preferred post structure:

1. `# Introduction`
   - Start from the statistical problem the paper studies.
   - Explain why the problem matters before listing metadata.

2. Concept / method sections with natural headings
   - Define the evaluation setup, estimand, estimator, uncertainty model, or calibration problem.
   - Derive central equations when useful.
   - Explain statistical components more deeply than AI benchmark details.

3. Figure / experiment interpretation sections
   - Include official paper figure captures only when available.
   - Explain what the reader should notice in the figure.
   - Do not simply restate the caption.

4. Short "Why this paper?" or "Research direction" section
   - Briefly mention why the paper was selected and what research directions it suggests.
   - Keep candidate ranking and rejected candidates out of the main body unless they add real value.

5. `# References`
   - DOI
   - arXiv ID
   - Semantic Scholar / OpenReview / official URL
   - PDF URL if available
   - Suggested tags
   - Manual reference-manager import metadata

Optional curation metadata:
- If useful, include a compact score table near the end, but do not let the post read like a scoring report.
- Rejected candidates may be omitted from the published post. Keep them in automation notes/memory instead if they make the article less readable.

Suggested tags:
- `llm-meets-statistics`
- `daily-paper`
- `causal-inference`
- `spatial-statistics`
- `time-series`
- `bayesian-optimization`
- `llm-evaluation`
- `vlm`
- `risk-aware-evaluation`
- `rag-evaluation`
- `statistical-inference`
- `uncertainty`

Use only tags that actually match the paper.

Important constraints:
- Do not hallucinate paper details.
- Every claim about the paper must be grounded in the abstract, metadata, official PDF, or reliable source.
- If a detail is unavailable, write:
  - "정보 확인 불가"
  - "not available"
- Keep the post concise, readable, and publication-ready.
- Avoid dense paragraphs.
- Prefer bullet lists, compact tables, and equations when they make the explanation clearer.
- Include links to all sources used.
- Clearly label your own research ideas as speculation or possible extensions.
- Do not present your own future research ideas as claims made by the paper.
- At the end, briefly state the exact metadata needed for manual reference-manager import:
  - title
  - authors
  - year
  - URL
  - DOI/arXiv ID
  - suggested tags
  - collection: `Recent Trends`

Validation workflow:
- After the MDX post and any local image assets are complete, review the generated files.
- Run:
  - npm run validate:content
- If broader publishing validation is needed, run:
  - npm run validate:publish
- Fix validation errors before committing.

Publish workflow:
- Commit only the new or changed daily post file and its related image assets.
- Do not commit unrelated local changes.
- Use a clear commit message:
  - Add daily research paper post for YYYY-MM-DD
- Push the commit to the current publishing branch:
  - master
- This repository's Pages deployment runs on pushes to master or main.
- If commit or push fails, report the exact failure and leave the generated post files in place.

Zotero workflow:
- Zotero local API is available only when running on the same local machine as Zotero Desktop.
- Do not assume Zotero local API will work inside GitHub Actions or another remote runner.
- After selecting the paper and creating the MDX post, connect to the Zotero local API at:
  - http://localhost:23119/api
  through the available Zotero plugin or connector.
- Confirm that Zotero Desktop is reachable and that the local API is enabled.
- Add the selected paper to the Zotero collection named:
  - Recent Trends
- If the `Recent Trends` collection does not exist, create it before adding the paper.
- Import only the selected paper, not the rejected candidates.
- Use reliable metadata from DOI, arXiv, Semantic Scholar, OpenReview, official proceedings, or the official PDF/source page.
- Attach the generated MDX post to the Zotero item if the Zotero connector supports local file attachments.
- If a paper PDF is reliably available and connector permissions allow it, attach or link the official PDF.
- Apply only tags that actually match the paper, including `llm-meets-statistics` and `daily-paper` when appropriate.
- Before creating a duplicate, search Zotero by title, DOI, and arXiv ID.
- If the item already exists, update its collection membership and tags instead of creating a duplicate.
- If Zotero access, collection creation, item import, or attachment upload fails, report the exact failure and provide the metadata needed for manual import into `Recent Trends`.
```
