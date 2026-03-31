export const SITE_URL = "https://ddangchani.github.io";
export const SOURCE_POSTS_DIR = "_posts";
export const CONTENT_POSTS_DIR = "content/posts";
export const GENERATED_CONTENT_DIR = "generated/content";
export const CONTENT_POSTS_GLOB = "content/posts/**/*.mdx";

export interface PostMeta {
  title: string;
  description: string;
  date: string;
  tags: string[];
  categories: string[];
  routeSegments: string[];
  teaser: string | null;
  coverImage: string | null;
  excerpt: string;
  readingTimeMinutes: number;
  useMath: boolean;
  featured: boolean;
  draft: boolean;
  canonicalUrl: string;
}

export interface PostDocument {
  meta: PostMeta;
  body: string;
  sourcePath: string;
}

export interface PostHeading {
  id: string;
  text: string;
  level: number;
}

export interface CompiledPostDocument extends PostDocument {
  html: string;
  headings: PostHeading[];
  route: string;
}

export interface SearchEntry {
  title: string;
  route: string;
  description: string;
  tags: string[];
  categories: string[];
  bodyPlain: string;
  date: string;
  language: "ko" | "en" | "mixed";
}

export interface PopularPostEntry {
  route: string;
  viewCount: number;
  title: string;
  teaser: string | null;
  excerpt: string;
}

export interface RouteManifestEntry {
  legacyId: string;
  title: string;
  date: string;
  routeSegments: string[];
  route: string;
  canonicalUrl: string;
  sourcePath: string;
}
