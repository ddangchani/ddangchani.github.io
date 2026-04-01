import fs from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import { spawnSync } from "node:child_process";

import yaml from "js-yaml";

import type { PostJobInput, PostJobResult } from "@/lib/publishing/contracts";
import {
  buildCanonicalUrl,
  buildLegacyRouteSegments,
  encodeRouteSegments
} from "@/lib/content/legacy-routes";
import { getAllPosts } from "@/lib/content/loaders";

type AgentMode = "plan" | "draft" | "publish";

type DraftPayload = {
  title: string;
  description: string;
  tags: string[];
  category: string | null;
  useMath: boolean;
  teaser: string | null;
  body: string;
};

type CliOptions = {
  mode: AgentMode;
  input: PostJobInput;
};

function fail(message: string): never {
  throw new Error(message);
}

function runCommand(command: string, args: string[], options?: { allowFailure?: boolean }) {
  const executable = process.platform === "win32" && command === "npm" ? "npm.cmd" : command;
  const result = spawnSync(executable, args, {
    stdio: "pipe",
    encoding: "utf8"
  });

  if (result.status !== 0 && !options?.allowFailure) {
    throw new Error(`${command} ${args.join(" ")} failed:\n${result.stderr || result.stdout}`);
  }

  return result;
}

function parseList(value: string | undefined): string[] {
  if (!value) {
    return [];
  }

  return value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function parseCli(): CliOptions {
  const parsed = parseArgs({
    allowPositionals: true,
    options: {
      topic: { type: "string" },
      audience: { type: "string" },
      tone: { type: "string" },
      language: { type: "string" },
      "source-urls": { type: "string" },
      "tag-hints": { type: "string" },
      publish: { type: "boolean" },
      "target-branch": { type: "string" }
    }
  });

  const mode = (parsed.positionals[0] ?? "draft") as AgentMode;

  if (!["plan", "draft", "publish"].includes(mode)) {
    fail("Mode must be one of: plan, draft, publish");
  }

  const topic = parsed.values.topic?.trim();

  if (!topic) {
    fail("--topic is required");
  }

  return {
    mode,
    input: {
      topic,
      audience: parsed.values.audience?.trim() || "engineers and technical readers",
      tone: parsed.values.tone?.trim() || "rigorous, practical, and clear",
      language: parsed.values.language?.trim() || "ko",
      sourceUrls: parseList(parsed.values["source-urls"]),
      tagHints: parseList(parsed.values["tag-hints"]),
      publish: Boolean(parsed.values.publish || mode === "publish"),
      targetBranch: parsed.values["target-branch"]?.trim() || "main"
    }
  };
}

function slugifyTopic(value: string): string {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{Script=Hangul}\p{Letter}\p{Number}\s-]+/gu, " ")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

function buildContentPostPath(date: string, slug: string): string {
  return path.join("content", "posts", "agentic", date.slice(0, 4), `${date}-${slug}.mdx`);
}

function normalizeTags(tagHints: string[], topic: string): string[] {
  const baseTags = tagHints.length > 0 ? tagHints : topic.split(/[\s/,-]+/).slice(0, 3);
  return [...new Set(baseTags.map((tag) => tag.trim()).filter(Boolean))].slice(0, 6);
}

function collectSiteContext() {
  const posts = getAllPosts();
  const recentPosts = [...posts]
    .sort((left, right) => right.meta.date.localeCompare(left.meta.date))
    .slice(0, 8)
    .map((post) => ({
      title: post.meta.title,
      route: encodeRouteSegments(post.meta.routeSegments),
      tags: post.meta.tags
    }));
  const tagCounts = new Map<string, number>();

  for (const post of posts) {
    for (const tag of post.meta.tags) {
      tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1);
    }
  }

  const topTags = [...tagCounts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0], "ko"))
    .slice(0, 10)
    .map(([tag]) => tag);

  return {
    recentPosts,
    topTags
  };
}

function extractJsonObject(value: string): string {
  const start = value.indexOf("{");
  const end = value.lastIndexOf("}");

  if (start === -1 || end === -1 || end <= start) {
    fail("Model output did not contain a JSON object");
  }

  return value.slice(start, end + 1);
}

async function requestDraftFromModel(input: PostJobInput, slug: string): Promise<DraftPayload | null> {
  const apiKey = process.env.OPENAI_API_KEY;
  const model = process.env.POST_AGENT_MODEL;

  if (!apiKey || !model) {
    return null;
  }

  const context = collectSiteContext();
  const prompt = [
    "You are drafting a technical blog post for a Git-based writing archive.",
    "Return valid JSON only with keys: title, description, tags, category, useMath, teaser, body.",
    "Do not wrap the JSON in markdown fences.",
    `Topic: ${input.topic}`,
    `Audience: ${input.audience}`,
    `Tone: ${input.tone}`,
    `Language: ${input.language}`,
    `Suggested slug: ${slug}`,
    `Source URLs: ${input.sourceUrls.join(", ") || "none provided"}`,
    `Tag hints: ${input.tagHints.join(", ") || "none"}`,
    `Recent posts: ${JSON.stringify(context.recentPosts)}`,
    `Top tags: ${JSON.stringify(context.topTags)}`,
    "Body requirements:",
    "- Start with a short setup section.",
    "- Use clear section headings.",
    "- Include a final section named 'References'.",
    "- If sources are provided, cite them as markdown links in the References section.",
    "- Keep claims conservative when sources are sparse.",
    "- Write markdown only inside the body string."
  ].join("\n");

  const response = await fetch(
    `${process.env.OPENAI_API_BASE_URL ?? "https://api.openai.com/v1"}/chat/completions`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content: "You produce strict JSON for a technical writing pipeline."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        response_format: { type: "json_object" }
      })
    }
  );

  if (!response.ok) {
    throw new Error(`Model request failed with ${response.status}: ${await response.text()}`);
  }

  const payload = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const content = payload.choices?.[0]?.message?.content;

  if (!content) {
    fail("Model response was empty");
  }

  return JSON.parse(extractJsonObject(content)) as DraftPayload;
}

function buildFallbackDraft(input: PostJobInput, slug: string): DraftPayload {
  const tags = normalizeTags(input.tagHints, input.topic);
  const references =
    input.sourceUrls.length > 0
      ? input.sourceUrls.map((url) => `- [${url}](${url})`).join("\n")
      : "- No external source URLs were provided for this draft.";

  return {
    title: input.topic,
    description: `Technical note on ${input.topic}.`,
    tags,
    category: null,
    useMath: false,
    teaser: "/assets/logos/background.png",
    body: [
      "## Setting",
      "",
      `${input.topic} is the focus of this draft. This version was generated from the local post agent scaffold and should be refined before final publication if stronger sourcing or deeper analysis is required.`,
      "",
      "## Key Ideas",
      "",
      `- Audience: ${input.audience}`,
      `- Tone: ${input.tone}`,
      `- Working slug: \`${slug}\``,
      "",
      "## Implementation Notes",
      "",
      "Add the concrete explanation, examples, and any supporting derivations here.",
      "",
      "## References",
      "",
      references
    ].join("\n")
  };
}

async function buildDraftPayload(input: PostJobInput, slug: string): Promise<DraftPayload> {
  const modelDraft = await requestDraftFromModel(input, slug);
  const fallback = buildFallbackDraft(input, slug);

  if (!modelDraft) {
    return fallback;
  }

  return {
    title: modelDraft.title?.trim() || fallback.title,
    description: modelDraft.description?.trim() || fallback.description,
    tags:
      Array.isArray(modelDraft.tags) && modelDraft.tags.length > 0
        ? modelDraft.tags.map((tag) => String(tag).trim()).filter(Boolean)
        : fallback.tags,
    category:
      typeof modelDraft.category === "string" && modelDraft.category.trim().length > 0
        ? modelDraft.category.trim()
        : null,
    useMath: Boolean(modelDraft.useMath),
    teaser:
      typeof modelDraft.teaser === "string" && modelDraft.teaser.trim().length > 0
        ? modelDraft.teaser.trim()
        : fallback.teaser,
    body: modelDraft.body?.trim() || fallback.body
  };
}

function createFrontmatter(payload: DraftPayload, date: string, slug: string): string {
  const routeSegments = buildLegacyRouteSegments(`${date}-${slug}.mdx`, payload.category);
  const frontmatter = {
    title: payload.title,
    description: payload.description,
    date,
    tags: payload.tags,
    categories: payload.category ? [payload.category] : [],
    routeSegments,
    teaser: payload.teaser,
    coverImage: payload.teaser,
    excerpt: payload.description,
    useMath: payload.useMath,
    featured: false,
    draft: false,
    canonicalUrl: buildCanonicalUrl(routeSegments)
  };

  return `---\n${yaml.dump(frontmatter, {
    lineWidth: 0,
    noRefs: true,
    quotingType: "\""
  })}---\n\n`;
}

function getPreexistingDirtyEntries(): string[] {
  return runCommand("git", ["status", "--porcelain"]).stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.slice(3));
}

function filterUnexpectedDirtyEntries(entries: string[], allowedPaths: string[]): string[] {
  return entries.filter(
    (entry) =>
      !allowedPaths.some((allowedPath) => entry === allowedPath || entry.startsWith(`${allowedPath}/`))
  );
}

function validateWorktreeIsSafe(beforeEntries: string[], allowedPaths: string[]) {
  const unexpectedEntries = filterUnexpectedDirtyEntries(beforeEntries, allowedPaths);

  if (unexpectedEntries.length > 0) {
    fail(
      `Publish blocked because the worktree has unrelated changes: ${unexpectedEntries.join(", ")}`
    );
  }
}

function writeDraftFile(filePath: string, draft: DraftPayload): void {
  const fileName = path.basename(filePath);
  const date = fileName.slice(0, 10);
  const slug = fileName.replace(/^\d{4}-\d{2}-\d{2}-/, "").replace(/\.mdx$/, "");
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${createFrontmatter(draft, date, slug)}${draft.body.trim()}\n`, "utf8");
}

function determineCreatedRoute(filePath: string, category: string | null): string {
  const routeSegments = buildLegacyRouteSegments(path.basename(filePath), category);
  return encodeRouteSegments(routeSegments);
}

function runPublishValidation(): void {
  runCommand("npm", ["run", "validate:publish"]);
}

function stageAndCommit(filePath: string, route: string): string {
  runCommand("git", ["add", filePath]);
  runCommand("git", ["commit", "-m", `feat(post): publish ${route}`]);
  return runCommand("git", ["rev-parse", "HEAD"]).stdout.trim();
}

function pushWithFallback(targetBranch: string): { pushed: boolean; failureReason: string | null } {
  const directPush = runCommand("git", ["push", "origin", `HEAD:${targetBranch}`], {
    allowFailure: true
  });

  if (directPush.status === 0) {
    return { pushed: true, failureReason: null };
  }

  const fallbackBranch = `codex/post-${new Date().toISOString().slice(0, 10)}-${Date.now()}`;
  runCommand("git", ["branch", fallbackBranch]);
  runCommand("git", ["push", "-u", "origin", fallbackBranch]);

  return {
    pushed: true,
    failureReason: `Direct push to ${targetBranch} failed. Pushed fallback branch ${fallbackBranch} instead.`
  };
}

async function execute() {
  const { mode, input } = parseCli();
  const date = new Date().toISOString().slice(0, 10);
  const slug = slugifyTopic(input.topic);

  if (!slug) {
    fail("Unable to derive a slug from the topic");
  }

  const filePath = buildContentPostPath(date, slug);
  const routePreview = determineCreatedRoute(filePath, null);
  const preexistingDirtyEntries = getPreexistingDirtyEntries();
  const safeDirtyAllowlist = [
    "content",
    "generated",
    "public",
    ".next",
    "out"
  ];

  if (mode === "plan") {
    const result: PostJobResult = {
      createdRoute: routePreview,
      contentFile: filePath,
      assetFiles: [],
      checksPassed: false,
      commitSha: null,
      pushed: false,
      failureReason: null
    };
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (fs.existsSync(filePath)) {
    fail(`Refusing to overwrite existing file: ${filePath}`);
  }

  if (mode === "publish") {
    validateWorktreeIsSafe(preexistingDirtyEntries, safeDirtyAllowlist);
  }

  const draft = await buildDraftPayload(input, slug);
  const route = determineCreatedRoute(filePath, draft.category);

  writeDraftFile(filePath, draft);

  const result: PostJobResult = {
    createdRoute: route,
    contentFile: filePath,
    assetFiles: [],
    checksPassed: false,
    commitSha: null,
    pushed: false,
    failureReason: null
  };

  if (mode === "draft") {
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  runPublishValidation();
  result.checksPassed = true;
  result.commitSha = stageAndCommit(filePath, route);

  const pushResult = pushWithFallback(input.targetBranch);
  result.pushed = pushResult.pushed;
  result.failureReason = pushResult.failureReason;

  console.log(JSON.stringify(result, null, 2));
}

execute().catch((error: unknown) => {
  const result: PostJobResult = {
    createdRoute: null,
    contentFile: null,
    assetFiles: [],
    checksPassed: false,
    commitSha: null,
    pushed: false,
    failureReason: error instanceof Error ? error.message : String(error)
  };

  console.error(JSON.stringify(result, null, 2));
  process.exitCode = 1;
});
