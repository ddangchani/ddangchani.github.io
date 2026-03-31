import path from "node:path";

import { SITE_URL } from "@/lib/content/contracts";

export function normalizeCategorySegment(category?: string | null): string | null {
  if (!category) {
    return null;
  }

  const normalized = category.trim().replace(/\s+/g, " ").toLowerCase();
  return normalized.length > 0 ? normalized : null;
}

export function buildLegacyId(fileName: string): string {
  return path.basename(fileName, path.extname(fileName));
}

export function buildLegacySlugFromFileName(fileName: string): string {
  const stem = buildLegacyId(fileName);
  return stem.replace(/^\d{4}-\d{2}-\d{2}-/, "");
}

export function buildLegacyRouteSegments(
  fileName: string,
  category?: string | null
): string[] {
  const categorySegment = normalizeCategorySegment(category);
  const slug = buildLegacySlugFromFileName(fileName);

  return [categorySegment, slug].filter((segment): segment is string => Boolean(segment));
}

export function encodeRouteSegments(routeSegments: string[]): string {
  return `/${routeSegments.map((segment) => encodeURIComponent(segment)).join("/")}/`;
}

export function buildCanonicalUrl(routeSegments: string[]): string {
  return new URL(encodeRouteSegments(routeSegments), SITE_URL).toString();
}

