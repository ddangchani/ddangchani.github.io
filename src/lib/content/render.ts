import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";

import { normalizeNewlines } from "@/lib/content/markdown";

function slugifyHeading(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function parseKramdownAttributes(attributeText: string): string {
  const classNames: string[] = [];
  const attributes: string[] = [];
  const tokens = attributeText.match(/(?:[^\s"]+|"[^"]*")+/g) ?? [];

  for (const token of tokens) {
    if (token.startsWith(".")) {
      classNames.push(token.slice(1));
      continue;
    }

    const [name, rawValue] = token.split("=");

    if (!name || rawValue == null) {
      continue;
    }

    const value = rawValue.replace(/^"|"$/g, "");
    attributes.push(`${name}="${value}"`);
  }

  if (classNames.length > 0) {
    attributes.unshift(`class="${classNames.join(" ")}"`);
  }

  return attributes.join(" ");
}

export function normalizeLegacyMarkdownForRendering(content: string): string {
  return normalizeNewlines(content).replace(
    /!\[([^\]]*)\]\(([^)]+)\)\{:\s*([^}]+)\}/g,
    (_match, alt: string, src: string, attributeText: string) => {
      const attributes = parseKramdownAttributes(attributeText);
      const attrBlock = attributes.length > 0 ? ` ${attributes}` : "";
      return `<img src="${src}" alt="${alt}"${attrBlock} />`;
    }
  );
}

export function extractHeadings(content: string): { id: string; text: string; level: number }[] {
  return normalizeNewlines(content)
    .split("\n")
    .map((line) => line.match(/^(#{1,6})\s+(.+?)\s*$/))
    .filter((match): match is RegExpMatchArray => Boolean(match))
    .map((match) => {
      const text = match[2].trim();
      return {
        id: slugifyHeading(text),
        text,
        level: match[1].length
      };
    });
}

export async function renderPostBodyToHtml(content: string): Promise<string> {
  const normalized = normalizeLegacyMarkdownForRendering(content);
  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype, {
      allowDangerousHtml: true
    })
    .use(rehypeRaw)
    .use(rehypeKatex, {
      strict: "ignore"
    })
    .use(rehypeSlug)
    .use(rehypeStringify, {
      allowDangerousHtml: true
    })
    .process(normalized);

  return String(result);
}
