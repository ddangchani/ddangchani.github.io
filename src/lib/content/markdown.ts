const HANGUL_PATTERN = /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]/;

export function normalizeNewlines(value: string): string {
  return value.replace(/\r\n/g, "\n");
}

export function stripCodeFences(value: string): string {
  return value.replace(/```[\s\S]*?```/g, " ");
}

export function stripMarkdown(value: string): string {
  const withoutCode = stripCodeFences(normalizeNewlines(value));

  return withoutCode
    .replace(/!\[[^\]]*]\(([^)]+)\)/g, " ")
    .replace(/\[([^\]]+)]\(([^)]+)\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/^>\s?/gm, "")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/[*_`~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function extractExcerpt(value: string, maxLength = 220): string {
  const stripped = stripMarkdown(value);

  if (stripped.length <= maxLength) {
    return stripped;
  }

  return `${stripped.slice(0, maxLength - 3).trimEnd()}...`;
}

export function estimateReadingTimeMinutes(value: string, wordsPerMinute = 200): number {
  const plainText = stripMarkdown(value);
  const words = plainText.length === 0 ? 0 : plainText.split(/\s+/).length;

  return Math.max(1, Math.ceil(words / wordsPerMinute));
}

export function detectLanguage(value: string): "ko" | "en" | "mixed" {
  const hasHangul = HANGUL_PATTERN.test(value);
  const hasLatin = /[A-Za-z]/.test(value);

  if (hasHangul && hasLatin) {
    return "mixed";
  }

  if (hasHangul) {
    return "ko";
  }

  return "en";
}
