import type {
  CompiledPostDocument,
  PopularPostEntry,
  SearchEntry
} from "@/lib/content/contracts";
import {
  loadCompiledPostByRouteSegments,
  loadCompiledPosts,
  loadPopularPosts,
  loadSearchIndex
} from "@/lib/content/loaders";

export type SitePost = CompiledPostDocument["meta"] & {
  route: string;
  body: string;
  html: string;
  headings: CompiledPostDocument["headings"];
  sourcePath: string;
};

export type SiteSearchEntry = SearchEntry;

export type SitePopularEntry = PopularPostEntry & {
  route: string;
};

function toSitePost(post: CompiledPostDocument): SitePost {
  return {
    ...post.meta,
    route: post.route,
    body: post.body,
    html: post.html,
    headings: post.headings,
    sourcePath: post.sourcePath
  };
}

function mergePopularPosts(popular: SitePopularEntry[], posts: SitePost[]): SitePopularEntry[] {
  const postIndex = new Map(posts.map((post) => [post.route, post]));

  return popular.map((entry) => {
    const match = postIndex.get(entry.route);

    if (!match) {
      return entry;
    }

    return {
      ...entry,
      title: entry.title || match.title,
      teaser: entry.teaser ?? match.teaser,
      excerpt: entry.excerpt || match.excerpt
    };
  });
}

export async function getAllPosts(): Promise<SitePost[]> {
  return loadCompiledPosts().map(toSitePost);
}

export async function getFeaturedPosts(): Promise<SitePost[]> {
  const posts = await getAllPosts();
  const featured = posts.filter((post) => post.featured);

  if (featured.length > 0) {
    return featured.slice(0, 3);
  }

  return posts.slice(0, 3);
}

export async function getRecentPosts(limit = 9): Promise<SitePost[]> {
  const posts = await getAllPosts();
  return posts.slice(0, limit);
}

export async function getAllRouteSegments(): Promise<string[][]> {
  const posts = await getAllPosts();
  return posts.map((post) => post.routeSegments);
}

export async function getPostBySegments(routeSegments: string[]): Promise<SitePost | null> {
  const post = loadCompiledPostByRouteSegments(routeSegments);
  return post ? toSitePost(post) : null;
}

export async function getAllTags(): Promise<string[]> {
  const posts = await getAllPosts();
  return [...new Set(posts.flatMap((post) => post.tags))].sort((left, right) =>
    left.localeCompare(right, "ko")
  );
}

export async function getAllCategories(): Promise<string[]> {
  const posts = await getAllPosts();
  return [...new Set(posts.flatMap((post) => post.categories))].sort((left, right) =>
    left.localeCompare(right, "ko")
  );
}

export async function getPostsByTag(tag: string): Promise<SitePost[]> {
  const posts = await getAllPosts();
  return posts.filter((post) => post.tags.some((item) => item.toLowerCase() === tag.toLowerCase()));
}

export async function getPostsByCategory(category: string): Promise<SitePost[]> {
  const posts = await getAllPosts();
  return posts.filter((post) =>
    post.categories.some((item) => item.toLowerCase() === category.toLowerCase())
  );
}

export async function getSearchEntries(): Promise<SiteSearchEntry[]> {
  try {
    return loadSearchIndex();
  } catch {
    return [];
  }
}

export async function getPopularEntries(): Promise<SitePopularEntry[]> {
  const [popular, posts] = await Promise.all([
    Promise.resolve()
      .then(() => loadPopularPosts())
      .catch(() => []),
    getAllPosts()
  ]);

  return mergePopularPosts(popular, posts);
}

export async function getTopicSpotlight(): Promise<Array<{ title: string; count: number }>> {
  const posts = await getAllPosts();
  const counts = new Map<string, number>();

  for (const post of posts) {
    for (const tag of post.tags) {
      counts.set(tag, (counts.get(tag) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .map(([title, count]) => ({ title, count }))
    .sort((left, right) => right.count - left.count || left.title.localeCompare(right.title, "ko"))
    .slice(0, 8);
}

export async function getTagSummaries(): Promise<Array<{ name: string; count: number }>> {
  const posts = await getAllPosts();
  const counts = new Map<string, number>();

  for (const post of posts) {
    for (const tag of post.tags) {
      counts.set(tag, (counts.get(tag) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((left, right) => right.count - left.count || left.name.localeCompare(right.name, "ko"));
}

export async function getCategorySummaries(): Promise<Array<{ name: string; count: number }>> {
  const posts = await getAllPosts();
  const counts = new Map<string, number>();

  for (const post of posts) {
    for (const category of post.categories) {
      counts.set(category, (counts.get(category) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((left, right) => right.count - left.count || left.name.localeCompare(right.name, "ko"));
}
