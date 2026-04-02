"use client";

import { clsx } from "clsx";
import { useEffect, useState } from "react";

import type { PostHeading } from "@/lib/content/contracts";

export const POST_PROSE_SENTINEL_ID = "post-prose-start";
export const POST_PROSE_END_SENTINEL_ID = "post-prose-end";

const DESKTOP_TOC_REVEAL_OFFSET_PX = 112;
const DESKTOP_TOC_ACTIVE_OFFSET_PX = 144;
const DESKTOP_TOC_BREAKPOINT_CLASS_NAME = "min-[1080px]:block";
const DESKTOP_TOC_TOP = "calc(4.65rem + 1rem)";
const DESKTOP_TOC_RIGHT =
  "max(var(--page-gutter), calc((100vw - min(1116px, calc(100vw - (var(--page-gutter) * 2)))) / 2))";
const DESKTOP_TOC_WIDTH = "12rem";

type PostTableOfContentsProps = {
  headings: PostHeading[];
};

type TocListProps = {
  headings: PostHeading[];
  variant: "full" | "compact";
  activeHeadingId?: string | null;
};

function getCompactHeadings(headings: PostHeading[]) {
  const preferredHeadings = headings.filter((heading) => heading.level === 2 || heading.level === 3);

  if (preferredHeadings.length > 0) {
    return preferredHeadings;
  }

  const fallbackHeadings = headings.filter((heading) => heading.level > 1);

  if (fallbackHeadings.length > 0) {
    return fallbackHeadings;
  }

  return headings;
}

function getItemClassName(level: number, variant: "full" | "compact") {
  if (variant === "compact") {
    if (level >= 3) {
      return "min-w-0 pl-3 text-[color:color-mix(in_srgb,var(--ink-soft)_92%,var(--ink)_8%)]";
    }

    return "min-w-0 text-[var(--ink-soft)]";
  }

  if (level >= 4) {
    return "min-w-0 pl-8 text-[var(--ink-soft)]";
  }

  if (level === 3) {
    return "min-w-0 pl-4 text-[var(--ink-soft)]";
  }

  return "min-w-0 text-[var(--ink-soft)]";
}

function TocList({ headings, variant, activeHeadingId }: TocListProps) {
  return (
    <ol
      className={clsx(
        "m-0 grid min-w-0 list-none p-0",
        variant === "full" ? "gap-2" : "gap-[0.35rem]"
      )}
    >
      {headings.map((heading) => {
        const isActive = heading.id === activeHeadingId;

        return (
          <li key={heading.id} className={getItemClassName(heading.level, variant)}>
            <a
              href={`#${heading.id}`}
              aria-current={isActive ? "location" : undefined}
              className={clsx(
                "block w-full [overflow-wrap:anywhere] transition-colors duration-200",
                variant === "full"
                  ? "hover:text-[var(--accent-strong)]"
                  : "rounded-[0.8rem] px-[0.55rem] py-[0.35rem] hover:text-[var(--ink)]",
                isActive
                  ? "bg-[color:color-mix(in_srgb,var(--accent)_10%,white)] text-[var(--accent-strong)]"
                  : variant === "compact"
                    ? "text-[var(--ink-soft)] hover:bg-[color:color-mix(in_srgb,var(--accent)_6%,white)]"
                    : "text-current"
              )}
            >
              {heading.text}
            </a>
          </li>
        );
      })}
    </ol>
  );
}

export function PostTableOfContents({ headings }: PostTableOfContentsProps) {
  if (headings.length === 0) {
    return null;
  }

  return (
    <aside
      className="grid min-w-0 gap-3 rounded-[1.5rem] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_92%,white)] px-5 py-4"
      aria-label="Table of Contents"
    >
      <p className="m-0 text-[0.78rem] uppercase tracking-[0.18em] text-[var(--ink-soft)]">
        Table of Contents
      </p>
      <TocList headings={headings} variant="full" />
    </aside>
  );
}

export function PostStickyTableOfContents({ headings }: PostTableOfContentsProps) {
  const compactHeadings = getCompactHeadings(headings);
  const [hasStartedReading, setHasStartedReading] = useState(false);
  const [hasReachedEnd, setHasReachedEnd] = useState(false);
  const [activeHeadingId, setActiveHeadingId] = useState<string | null>(compactHeadings[0]?.id ?? null);

  useEffect(() => {
    const trackedHeadings = getCompactHeadings(headings);

    if (trackedHeadings.length === 0) {
      return;
    }

    const sentinel = document.getElementById(POST_PROSE_SENTINEL_ID);

    if (!sentinel) {
      return;
    }

    const revealObserver = new IntersectionObserver(
      ([entry]) => {
        setHasStartedReading(!entry.isIntersecting);
      },
      {
        rootMargin: `-${DESKTOP_TOC_REVEAL_OFFSET_PX}px 0px 0px 0px`,
        threshold: 0
      }
    );

    revealObserver.observe(sentinel);

    return () => {
      revealObserver.disconnect();
    };
  }, [headings]);

  useEffect(() => {
    const endSentinel = document.getElementById(POST_PROSE_END_SENTINEL_ID);

    if (!endSentinel) {
      return;
    }

    const endObserver = new IntersectionObserver(
      ([entry]) => {
        setHasReachedEnd(entry.isIntersecting);
      },
      {
        rootMargin: `-${DESKTOP_TOC_ACTIVE_OFFSET_PX}px 0px 0px 0px`,
        threshold: 0
      }
    );

    endObserver.observe(endSentinel);

    return () => {
      endObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    const trackedHeadings = getCompactHeadings(headings);

    if (trackedHeadings.length === 0) {
      return;
    }

    const targets = trackedHeadings
      .map((heading) => document.getElementById(heading.id))
      .filter((element): element is HTMLElement => element instanceof HTMLElement);

    if (targets.length === 0) {
      return;
    }

    const visibleHeadingIds = new Set<string>();

    const activeObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const headingId = (entry.target as HTMLElement).id;

          if (entry.isIntersecting) {
            visibleHeadingIds.add(headingId);
          } else {
            visibleHeadingIds.delete(headingId);
          }
        }

        setActiveHeadingId((current) => {
          let nextHeadingId = current ?? trackedHeadings[0]?.id ?? null;

          for (const heading of trackedHeadings) {
            if (visibleHeadingIds.has(heading.id)) {
              nextHeadingId = heading.id;
            }
          }

          return nextHeadingId;
        });
      },
      {
        rootMargin: `-${DESKTOP_TOC_ACTIVE_OFFSET_PX}px 0px -55% 0px`,
        threshold: [0, 1]
      }
    );

    for (const target of targets) {
      activeObserver.observe(target);
    }

    return () => {
      activeObserver.disconnect();
    };
  }, [headings]);

  if (compactHeadings.length === 0) {
    return null;
  }

  const isVisible = hasStartedReading && !hasReachedEnd;

  return (
    <aside className={clsx("hidden", DESKTOP_TOC_BREAKPOINT_CLASS_NAME)} aria-label="Current section navigation">
      <div
        className={clsx(
          "fixed grid max-h-[calc(100vh-7rem)] gap-3 overflow-y-auto rounded-[1.25rem] border border-[color:color-mix(in_srgb,var(--accent-strong)_10%,var(--line))] bg-[color:color-mix(in_srgb,var(--surface)_94%,white)] px-3 py-3 shadow-[0_18px_40px_color-mix(in_srgb,var(--accent-strong)_8%,transparent)] transition duration-300 ease-out motion-reduce:transform-none motion-reduce:transition-none",
          isVisible ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-2 opacity-0"
        )}
        style={{ top: DESKTOP_TOC_TOP, right: DESKTOP_TOC_RIGHT, width: DESKTOP_TOC_WIDTH }}
      >
        <p className="m-0 text-[0.68rem] uppercase tracking-[0.18em] text-[var(--ink-soft)]">
          Contents
        </p>
        <TocList
          headings={compactHeadings}
          variant="compact"
          activeHeadingId={activeHeadingId ?? compactHeadings[0]?.id ?? null}
        />
      </div>
    </aside>
  );
}
