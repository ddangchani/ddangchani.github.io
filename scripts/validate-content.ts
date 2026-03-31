import fs from "node:fs";
import path from "node:path";

import { CONTENT_POSTS_DIR } from "@/lib/content/contracts";
import { encodeRouteSegments } from "@/lib/content/legacy-routes";
import { getAllPosts } from "@/lib/content/loaders";
import { stripCodeFences } from "@/lib/content/markdown";

function collectAssetReferences(content: string): string[] {
  const matches = Array.from(
    content.matchAll(/(?:^|[\s(="'`])(?<asset>\/assets\/[^)\n\r"'<>]+)(?=[)"'<>\n\r])/gm),
    (match) => match.groups?.asset
  ).filter((asset): asset is string => Boolean(asset));

  return [...new Set(matches)];
}

function assertLocalAssetsExist(content: string, sourcePath: string): void {
  for (const assetReference of collectAssetReferences(stripCodeFences(content))) {
    const decodedPath = decodeURIComponent(assetReference);
    const localPath = path.join(process.cwd(), decodedPath.replace(/^\//, ""));

    if (!fs.existsSync(localPath)) {
      throw new Error(`Missing asset "${assetReference}" referenced by ${sourcePath}`);
    }
  }
}

function assertNoUnescapedLiquid(content: string, sourcePath: string): void {
  const withoutCode = stripCodeFences(content);

  if (/(?<!\\)\{[%{]/.test(withoutCode)) {
    throw new Error(`Unescaped Liquid token remains in ${sourcePath}`);
  }
}

function assertNoReplacementCharacters(content: string, sourcePath: string): void {
  if (content.includes("�")) {
    throw new Error(`Replacement character detected in ${sourcePath}`);
  }
}

function main(): void {
  const posts = getAllPosts(CONTENT_POSTS_DIR);
  const seenRoutes = new Map<string, string>();

  for (const post of posts) {
    const route = encodeRouteSegments(post.meta.routeSegments);

    if (seenRoutes.has(route)) {
      throw new Error(`Duplicate route ${route} between ${seenRoutes.get(route)} and ${post.sourcePath}`);
    }

    seenRoutes.set(route, post.sourcePath);

    if (post.body.trim().length === 0) {
      throw new Error(`Empty content body in ${post.sourcePath}`);
    }

    assertLocalAssetsExist(post.body, post.sourcePath);
    assertNoUnescapedLiquid(post.body, post.sourcePath);
    assertNoReplacementCharacters(post.body, post.sourcePath);
  }

  console.log(`Validated ${posts.length} migrated content files`);
}

main();
