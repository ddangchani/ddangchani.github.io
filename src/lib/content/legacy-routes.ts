import crypto from "node:crypto";
import path from "node:path";

import { SITE_URL } from "@/lib/content/contracts";

export function normalizeLegacyText(value: string): string {
  return value.normalize("NFC");
}

export function normalizeCategorySegment(category?: string | null): string | null {
  if (!category) {
    return null;
  }

  const normalized = category.trim().replace(/\s+/g, " ").toLowerCase();
  return normalized.length > 0 ? normalized : null;
}

export function buildLegacyId(fileName: string): string {
  return normalizeLegacyText(path.basename(fileName, path.extname(fileName)));
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

function buildShortHash(value: string): string {
  return crypto.createHash("sha1").update(normalizeLegacyText(value)).digest("hex").slice(0, 8);
}

function slugifySegment(value: string): string {
  return decodeRouteSegment(value)
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function buildPublicPostRouteSegments(fileName: string): string[] {
  const preferredSlug = slugifySegment(buildLegacySlugFromFileName(fileName));

  if (preferredSlug.length >= 6) {
    return [preferredSlug];
  }

  if (preferredSlug.length > 0) {
    return [`${preferredSlug}-${buildShortHash(buildLegacyId(fileName)).slice(0, 6)}`];
  }

  return [`entry-${buildShortHash(buildLegacyId(fileName))}`];
}

export function buildPublicPostRoutePath(fileName: string): string {
  return buildRoutePath(buildPublicPostRouteSegments(fileName));
}

export function decodeRouteSegment(segment: string): string {
  try {
    return normalizeLegacyText(decodeURIComponent(segment));
  } catch {
    return normalizeLegacyText(segment);
  }
}

export function encodeRouteSegment(segment: string): string {
  return encodeURIComponent(decodeRouteSegment(segment));
}

export function normalizeRouteSegments(routeSegments: string[]): string[] {
  return routeSegments
    .map((segment) => decodeRouteSegment(segment))
    .filter((segment) => segment.length > 0);
}

export function buildRoutePath(routeSegments: string[]): string {
  const normalizedSegments = normalizeRouteSegments(routeSegments);
  return `/${normalizedSegments.join("/")}/`;
}

export function encodeRouteSegments(routeSegments: string[]): string {
  return `/${normalizeRouteSegments(routeSegments).map((segment) => encodeRouteSegment(segment)).join("/")}/`;
}

export function buildCanonicalUrl(routeSegments: string[]): string {
  return new URL(encodeRouteSegments(routeSegments), SITE_URL).toString();
}
