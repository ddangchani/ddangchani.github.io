import fs from "node:fs";
import path from "node:path";

import fg from "fast-glob";
import yaml from "js-yaml";

import { CONTENT_POSTS_DIR } from "@/lib/content/contracts";
import { buildCanonicalUrl } from "@/lib/content/legacy-routes";
import {
  estimateReadingTimeMinutes,
  extractExcerpt,
  normalizeNewlines
} from "@/lib/content/markdown";
import { readSourcePosts } from "@/lib/content/source-posts";

function ensureDirectory(targetPath: string): void {
  fs.mkdirSync(targetPath, { recursive: true });
}

const knownAssetPaths = fg
  .sync("assets/**/*.*", {
    onlyFiles: true,
    dot: false
  })
  .map((assetPath) => `/${assetPath.replace(/\\/g, "/")}`)
  .sort((left, right) => right.length - left.length);

function stripOuterRawTags(content: string): string {
  return content
    .replace(/^\s*\{%\s*raw\s*%\}\s*\n?/, "")
    .replace(/\n?\s*\{%\s*endraw\s*%\}\s*$/, "")
    .trim();
}

function encodeLocalAssetPath(assetPath: string): string {
  if (!assetPath.startsWith("/assets/")) {
    return assetPath;
  }

  const strictEncodeSegment = (segment: string): string =>
    encodeURIComponent(segment).replace(/[!'()*]/g, (character) =>
      `%${character.charCodeAt(0).toString(16).toUpperCase()}`
    );

  try {
    const decoded = decodeURI(assetPath);
    return decoded
      .split("/")
      .map((segment, index) => (index === 0 ? segment : strictEncodeSegment(segment)))
      .join("/");
  } catch {
    return assetPath
      .split("/")
      .map((segment, index) => (index === 0 ? segment : strictEncodeSegment(segment)))
      .join("/");
  }
}

function encodeLocalAssetPaths(content: string): string {
  return content.replace(/\/assets\/[^)\n\r"'<>]+(?=[)"'<>\n\r])/g, (assetPath) =>
    encodeLocalAssetPath(assetPath)
  );
}

function replaceKnownAssetPaths(content: string): string {
  let normalized = content;

  for (const assetPath of knownAssetPaths) {
    if (normalized.includes(assetPath)) {
      normalized = normalized.split(assetPath).join(encodeLocalAssetPath(assetPath));
    }
  }

  return normalized;
}

function escapeLiquidOutsideCodeFences(content: string, routeLookup: Map<string, string>): string {
  const lines = normalizeNewlines(content).split("\n");
  let inFence = false;

  return lines
    .map((line) => {
      if (/^\s*```/.test(line)) {
        inFence = !inFence;
        return line;
      }

      if (inFence) {
        return line;
      }

      const withResolvedPostUrls = line.replace(
        /\{%\s*post_url\s+([^\s%]+)\s*%\}/g,
        (match, legacyId: string) => routeLookup.get(legacyId) ?? `\\${match}`
      );

      return withResolvedPostUrls.replace(/\{(?=[{%])/g, "\\{");
    })
    .join("\n")
    .trim()
    .concat("\n");
}

function normalizeFrontmatter(postBody: string, sourcePost: ReturnType<typeof readSourcePosts>[number]) {
  const excerpt = extractExcerpt(postBody);
  const teaser = sourcePost.teaser ? encodeLocalAssetPath(sourcePost.teaser) : null;

  return {
    title: sourcePost.title,
    description: excerpt,
    date: sourcePost.date,
    tags: sourcePost.tags,
    categories: sourcePost.categories,
    routeSegments: sourcePost.routeSegments,
    teaser,
    coverImage: teaser,
    excerpt,
    readingTimeMinutes: estimateReadingTimeMinutes(postBody),
    useMath: sourcePost.useMath,
    featured: false,
    draft: false,
    canonicalUrl: buildCanonicalUrl(sourcePost.routeSegments)
  };
}

function createFrontmatterBlock(frontmatter: Record<string, unknown>): string {
  return `---\n${yaml.dump(frontmatter, {
    lineWidth: 0,
    noRefs: true,
    quotingType: "\""
  })}---\n\n`;
}

function destinationPathFor(sourceFileName: string): string {
  const year = sourceFileName.slice(0, 4);
  return path.join(CONTENT_POSTS_DIR, year, sourceFileName.replace(/\.md$/, ".mdx"));
}

function main(): void {
  const sourcePosts = readSourcePosts();
  const routeLookup = new Map(sourcePosts.map((post) => [post.legacyId, post.route]));

  for (const sourcePost of sourcePosts) {
    const normalizedBody = encodeLocalAssetPaths(
      replaceKnownAssetPaths(
        escapeLiquidOutsideCodeFences(stripOuterRawTags(sourcePost.content), routeLookup)
      )
    );
    const frontmatter = normalizeFrontmatter(normalizedBody, sourcePost);
    const destinationPath = destinationPathFor(sourcePost.sourceFileName);

    ensureDirectory(path.dirname(destinationPath));
    fs.writeFileSync(
      destinationPath,
      `${createFrontmatterBlock(frontmatter)}${normalizedBody}`,
      "utf8"
    );
  }

  console.log(`Migrated ${sourcePosts.length} posts into ${CONTENT_POSTS_DIR}`);
}

main();
