import fs from "node:fs";
import path from "node:path";

import { GENERATED_CONTENT_DIR } from "@/lib/content/contracts";
import { encodeRouteSegments } from "@/lib/content/legacy-routes";
import { getAllPosts } from "@/lib/content/loaders";
import { getSourceRouteManifest } from "@/lib/content/source-posts";

function ensureDirectory(targetPath: string): void {
  fs.mkdirSync(targetPath, { recursive: true });
}

function main(): void {
  const sourceManifest = getSourceRouteManifest().sort((left, right) =>
    left.legacyId.localeCompare(right.legacyId)
  );
  const contentManifest = getAllPosts()
    .map((post) => ({
      legacyId: path.basename(post.sourcePath, path.extname(post.sourcePath)),
      route: encodeRouteSegments(post.meta.routeSegments),
      routeSegments: post.meta.routeSegments
    }))
    .sort((left, right) => left.legacyId.localeCompare(right.legacyId));

  if (sourceManifest.length !== contentManifest.length) {
    throw new Error(
      `Route parity count mismatch: source=${sourceManifest.length} content=${contentManifest.length}`
    );
  }

  for (let index = 0; index < sourceManifest.length; index += 1) {
    const source = sourceManifest[index];
    const content = contentManifest[index];

    if (source.legacyId !== content.legacyId) {
      throw new Error(`Legacy id mismatch at index ${index}: ${source.legacyId} vs ${content.legacyId}`);
    }

    if (source.route !== content.route) {
      throw new Error(`Route mismatch for ${source.legacyId}: ${source.route} vs ${content.route}`);
    }
  }

  ensureDirectory(GENERATED_CONTENT_DIR);
  fs.writeFileSync(
    path.join(GENERATED_CONTENT_DIR, "source-route-manifest.json"),
    `${JSON.stringify(sourceManifest, null, 2)}\n`,
    "utf8"
  );

  console.log(`Route parity passed for ${sourceManifest.length} posts`);
}

main();
