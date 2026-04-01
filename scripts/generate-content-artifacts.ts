import fs from "node:fs";
import path from "node:path";

import MiniSearch from "minisearch";

import {
  type CompiledPostDocument,
  GENERATED_CONTENT_DIR,
  SITE_URL,
  type PopularPostEntry,
  type RouteManifestEntry,
  type SearchEntry
} from "@/lib/content/contracts";
import {
  buildLegacyId,
  buildPublicPostRoutePath,
  encodeRouteSegments
} from "@/lib/content/legacy-routes";
import { detectLanguage, stripMarkdown } from "@/lib/content/markdown";
import { getAllPosts } from "@/lib/content/loaders";
import { extractHeadings, renderPostBodyToHtml } from "@/lib/content/render";

function ensureDirectory(targetPath: string): void {
  fs.mkdirSync(targetPath, { recursive: true });
}

function writeTextArtifact(fileName: string, value: string): void {
  ensureDirectory(GENERATED_CONTENT_DIR);
  fs.writeFileSync(path.join(GENERATED_CONTENT_DIR, fileName), value, "utf8");
}

function writeJsonArtifact(fileName: string, value: unknown): void {
  writeTextArtifact(fileName, `${JSON.stringify(value, null, 2)}\n`);
}

function buildRouteManifest(): RouteManifestEntry[] {
  return getAllPosts().map((post) => ({
    legacyId: buildLegacyId(post.sourcePath),
    title: post.meta.title,
    date: post.meta.date,
    routeSegments: post.meta.routeSegments,
    route: buildPublicPostRoutePath(post.sourcePath),
    canonicalUrl: post.meta.canonicalUrl,
    sourcePath: post.sourcePath
  }));
}

function buildSearchIndex(): SearchEntry[] {
  return getAllPosts().map((post) => {
    const bodyPlain = stripMarkdown(post.body);

    return {
      title: post.meta.title,
      route: buildPublicPostRoutePath(post.sourcePath),
      description: post.meta.description,
      tags: post.meta.tags,
      categories: post.meta.categories,
      bodyPlain,
      date: post.meta.date,
      language: detectLanguage(`${post.meta.title} ${post.meta.description} ${bodyPlain}`)
    };
  });
}

function buildSearchCorpus(entries: SearchEntry[]): ReturnType<MiniSearch<SearchEntry & { id: string }>["toJSON"]> {
  const miniSearch = new MiniSearch<SearchEntry & { id: string }>({
    idField: "id",
    fields: ["title", "description", "bodyPlain", "tags", "categories"],
    storeFields: ["title", "route", "description", "tags", "categories", "date", "language"],
    searchOptions: {
      prefix: true,
      fuzzy: 0.2,
      boost: {
        title: 4,
        tags: 3,
        description: 2
      }
    }
  });

  miniSearch.addAll(
    entries.map((entry) => ({
      ...entry,
      id: entry.route
    }))
  );
  return miniSearch.toJSON();
}

async function buildCompiledPosts(): Promise<CompiledPostDocument[]> {
  const posts = getAllPosts();

  return Promise.all(
    posts.map(async (post) => ({
      ...post,
      html: await renderPostBodyToHtml(post.body),
      headings: extractHeadings(post.body),
      route: buildPublicPostRoutePath(post.sourcePath)
    }))
  );
}

function buildPopularPosts(): PopularPostEntry[] {
  const analyticsPath = path.join("_data", "analytics.json");

  if (!fs.existsSync(analyticsPath)) {
    return [];
  }

  const analytics = JSON.parse(fs.readFileSync(analyticsPath, "utf8")) as Record<
    string,
    { count: number }
  >;
  const postsByRoute = new Map(
    getAllPosts().map((post) => [encodeRouteSegments(post.meta.routeSegments), post])
  );

  return Object.entries(analytics)
    .map(([route, entry]) => {
      const post = postsByRoute.get(route);
      if (!post) {
        return null;
      }

      return {
        route: buildPublicPostRoutePath(post.sourcePath),
        viewCount: entry.count,
        title: post.meta.title,
        teaser: post.meta.teaser,
        excerpt: post.meta.excerpt
      } satisfies PopularPostEntry;
    })
    .filter((entry): entry is PopularPostEntry => Boolean(entry))
    .sort((left, right) => right.viewCount - left.viewCount);
}

function xmlEscape(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function buildFeedXml(): string {
  const items = getAllPosts()
    .slice(0, 50)
    .map((post) => {
      const route = encodeRouteSegments(post.meta.routeSegments);
      const link = new URL(route, SITE_URL).toString();

      return `  <item>
    <title>${xmlEscape(post.meta.title)}</title>
    <link>${link}</link>
    <guid>${link}</guid>
    <pubDate>${new Date(post.meta.date).toUTCString()}</pubDate>
    <description>${xmlEscape(post.meta.description)}</description>
  </item>`;
    })
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>DDangchani's DataLog</title>
  <link>${SITE_URL}</link>
  <description>Editorial technical research blog.</description>
${items}
</channel>
</rss>
`;
}

function buildSitemapXml(): string {
  const urls = getAllPosts()
    .map((post) => {
      const route = encodeRouteSegments(post.meta.routeSegments);
      return `  <url>
    <loc>${new URL(route, SITE_URL).toString()}</loc>
    <lastmod>${post.meta.date}</lastmod>
  </url>`;
    })
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls}
</urlset>
`;
}

function buildRobotsTxt(): string {
  return `User-agent: *\nAllow: /\nSitemap: ${new URL("/sitemap.xml", SITE_URL).toString()}\n`;
}

async function main(): Promise<void> {
  const routeManifest = buildRouteManifest();
  const searchIndex = buildSearchIndex();
  const popularPosts = buildPopularPosts();
  const compiledPosts = await buildCompiledPosts();

  writeJsonArtifact("route-manifest.json", routeManifest);
  writeJsonArtifact("compiled-posts.json", compiledPosts);
  writeJsonArtifact("search-index.json", searchIndex);
  writeJsonArtifact("search-corpus.json", buildSearchCorpus(searchIndex));
  writeJsonArtifact("popular-posts.json", popularPosts);
  writeJsonArtifact(
    "tag-manifest.json",
    Array.from(
      searchIndex.reduce((map, entry) => {
        for (const tag of entry.tags) {
          const routes = map.get(tag) ?? [];
          routes.push(entry.route);
          map.set(tag, routes);
        }
        return map;
      }, new Map<string, string[]>())
    )
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([tag, routes]) => ({ tag, routes }))
  );
  writeTextArtifact("feed.xml", buildFeedXml());
  writeTextArtifact("sitemap.xml", buildSitemapXml());
  writeTextArtifact("robots.txt", buildRobotsTxt());

  console.log(
    `Generated ${routeManifest.length} routes, ${searchIndex.length} search entries, and ${popularPosts.length} popular post entries`
  );
}

void main();
