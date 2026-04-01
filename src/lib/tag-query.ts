export function encodeTagQueryValue(tag: string): string {
  return tag.replaceAll(" ", "_");
}

export function decodeTagQueryValue(value: string): string {
  try {
    return decodeURIComponent(value).replaceAll("_", " ");
  } catch {
    return value.replaceAll("_", " ");
  }
}

export function normalizeTagQueryValue(value: string): string {
  return decodeTagQueryValue(value).trim().toLocaleLowerCase();
}

export function tagsMatch(tag: string, queryValue: string): boolean {
  return normalizeTagQueryValue(tag) === normalizeTagQueryValue(queryValue);
}

export function buildTagFilterHref(tag: string): string {
  return `/posts/?tag=${encodeURIComponent(encodeTagQueryValue(tag))}`;
}
