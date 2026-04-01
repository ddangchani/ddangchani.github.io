import type { SiteSearchEntry } from "@/lib/site-data";

export function matchesSearchEntry(entry: SiteSearchEntry, query: string): boolean {
  const haystack = [
    entry.title,
    entry.description,
    entry.bodyPlain,
    entry.tags.join(" "),
    entry.categories.join(" ")
  ]
    .join(" ")
    .toLocaleLowerCase();

  return haystack.includes(query.toLocaleLowerCase());
}

export function filterSearchEntries(
  entries: SiteSearchEntry[],
  query: string,
  limit: number
): SiteSearchEntry[] {
  const normalizedQuery = query.trim();

  if (!normalizedQuery) {
    return entries.slice(0, limit);
  }

  return entries.filter((entry) => matchesSearchEntry(entry, normalizedQuery)).slice(0, limit);
}
