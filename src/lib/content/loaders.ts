import fs from "node:fs";
import path from "node:path";

import fg from "fast-glob";
import matter from "gray-matter";

import {
  type CompiledPostDocument,
  CONTENT_POSTS_DIR,
  GENERATED_CONTENT_DIR,
  type PopularPostEntry,
  type PostDocument,
  type RouteManifestEntry,
  type SearchEntry
} from "@/lib/content/contracts";
import { encodeRouteSegments } from "@/lib/content/legacy-routes";
import {
  popularPostEntrySchema,
  postMetaSchema,
  searchEntrySchema
} from "@/lib/content/schema";

export function getAllPostSourcePaths(contentDir = CONTENT_POSTS_DIR): string[] {
  return fg.sync("**/*.mdx", {
    cwd: contentDir,
    onlyFiles: true
  });
}

export function parsePostFile(relativePath: string, contentDir = CONTENT_POSTS_DIR): PostDocument {
  const sourcePath = path.join(contentDir, relativePath);
  const raw = fs.readFileSync(sourcePath, "utf8");
  const parsed = matter(raw);
  const meta = postMetaSchema.parse(parsed.data);

  return {
    meta,
    body: parsed.content,
    sourcePath
  };
}

export function getAllPosts(contentDir = CONTENT_POSTS_DIR): PostDocument[] {
  return getAllPostSourcePaths(contentDir)
    .map((relativePath) => parsePostFile(relativePath, contentDir))
    .sort((left, right) => right.meta.date.localeCompare(left.meta.date));
}

export function getPostByRouteSegments(routeSegments: string[]): PostDocument | null {
  return (
    getAllPosts().find(
      (post) => encodeRouteSegments(post.meta.routeSegments) === encodeRouteSegments(routeSegments)
    ) ?? null
  );
}

function readGeneratedJson<T>(fileName: string): T {
  const targetPath = path.join(GENERATED_CONTENT_DIR, fileName);
  return JSON.parse(fs.readFileSync(targetPath, "utf8")) as T;
}

export function loadSearchIndex(): SearchEntry[] {
  return readGeneratedJson<unknown[]>("search-index.json").map((entry) =>
    searchEntrySchema.parse(entry)
  );
}

export function loadPopularPosts(): PopularPostEntry[] {
  return readGeneratedJson<unknown[]>("popular-posts.json").map((entry) =>
    popularPostEntrySchema.parse(entry)
  );
}

export function loadRouteManifest(): RouteManifestEntry[] {
  return readGeneratedJson<RouteManifestEntry[]>("route-manifest.json");
}

export function loadCompiledPosts(): CompiledPostDocument[] {
  return readGeneratedJson<CompiledPostDocument[]>("compiled-posts.json");
}

export function loadCompiledPostByRouteSegments(
  routeSegments: string[]
): CompiledPostDocument | null {
  return (
    loadCompiledPosts().find(
      (post) => encodeRouteSegments(post.meta.routeSegments) === encodeRouteSegments(routeSegments)
    ) ?? null
  );
}
