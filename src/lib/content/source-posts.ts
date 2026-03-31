import fs from "node:fs";
import path from "node:path";

import fg from "fast-glob";
import matter from "gray-matter";

import {
  SOURCE_POSTS_DIR,
  type RouteManifestEntry
} from "@/lib/content/contracts";
import {
  buildCanonicalUrl,
  buildLegacyId,
  buildLegacyRouteSegments,
  encodeRouteSegments
} from "@/lib/content/legacy-routes";

export interface SourcePostRecord {
  sourcePath: string;
  sourceFileName: string;
  legacyId: string;
  title: string;
  date: string;
  tags: string[];
  categories: string[];
  routeSegments: string[];
  route: string;
  teaser: string | null;
  useMath: boolean;
  content: string;
  frontmatter: Record<string, unknown>;
}

function normalizeStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((entry) => String(entry).trim())
      .filter((entry) => entry.length > 0);
  }

  if (typeof value === "string" && value.trim().length > 0) {
    return [value.trim()];
  }

  return [];
}

function normalizeCategory(frontmatter: Record<string, unknown>): string[] {
  if (Array.isArray(frontmatter.categories)) {
    return normalizeStringList(frontmatter.categories);
  }

  if (frontmatter.category != null) {
    return normalizeStringList(frontmatter.category);
  }

  return [];
}

export function readSourcePosts(sourceDir = SOURCE_POSTS_DIR): SourcePostRecord[] {
  const files = fg.sync("*.md", {
    cwd: sourceDir,
    onlyFiles: true
  });

  return files
    .map((fileName) => {
      const sourcePath = path.join(sourceDir, fileName);
      const raw = fs.readFileSync(sourcePath, "utf8");
      const parsed = matter(raw);
      const frontmatter = parsed.data as Record<string, unknown>;
      const categories = normalizeCategory(frontmatter);
      const routeSegments = buildLegacyRouteSegments(fileName, categories[0] ?? null);
      const title =
        typeof frontmatter.title === "string" && frontmatter.title.trim().length > 0
          ? frontmatter.title.trim()
          : buildLegacyId(fileName);
      const date = fileName.slice(0, 10);
      const header =
        frontmatter.header && typeof frontmatter.header === "object"
          ? (frontmatter.header as Record<string, unknown>)
          : {};
      const teaser =
        typeof header.teaser === "string" && header.teaser.trim().length > 0
          ? header.teaser.trim()
          : null;

      return {
        sourcePath,
        sourceFileName: fileName,
        legacyId: buildLegacyId(fileName),
        title,
        date,
        tags: normalizeStringList(frontmatter.tags),
        categories,
        routeSegments,
        route: encodeRouteSegments(routeSegments),
        teaser,
        useMath: Boolean(frontmatter.use_math),
        content: parsed.content,
        frontmatter
      };
    })
    .sort((left, right) => left.date.localeCompare(right.date));
}

export function getSourceRouteManifest(): RouteManifestEntry[] {
  return readSourcePosts().map((post) => ({
    legacyId: post.legacyId,
    title: post.title,
    date: post.date,
    routeSegments: post.routeSegments,
    route: post.route,
    canonicalUrl: buildCanonicalUrl(post.routeSegments),
    sourcePath: post.sourcePath
  }));
}

