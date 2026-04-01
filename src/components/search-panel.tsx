"use client";

import { clsx } from "clsx";
import Link from "next/link";
import { useDeferredValue, useState } from "react";

import { filterSearchEntries } from "@/lib/search";
import type { SiteSearchEntry } from "@/lib/site-data";

type SearchPanelProps = {
  entries: SiteSearchEntry[];
  variant?: "page" | "compact";
  autoFocus?: boolean;
  inputId?: string;
  onNavigate?: () => void;
  className?: string;
};

export function SearchPanel({
  entries,
  variant = "page",
  autoFocus = false,
  inputId = "search-query",
  onNavigate,
  className
}: SearchPanelProps) {
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query.trim());
  const results = filterSearchEntries(entries, deferredQuery, variant === "compact" ? 8 : 18);
  const hasQuery = deferredQuery.length > 0;
  const initialCount = variant === "compact" ? 6 : 12;
  const visibleResults = hasQuery ? results : entries.slice(0, initialCount);
  const isCompact = variant === "compact";

  return (
    <section
      className={clsx(
        "grid rounded-[var(--radius-lg)] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_90%,white)] px-[clamp(1.2rem,3vw,2rem)] py-[clamp(1.2rem,3vw,2rem)] shadow-[var(--shadow)] max-[720px]:p-4",
        isCompact &&
          "gap-[0.8rem] rounded-[1.55rem] bg-[linear-gradient(180deg,color-mix(in_srgb,white_82%,var(--surface)_18%),color-mix(in_srgb,white_94%,var(--surface)_6%))] p-4",
        !isCompact && "gap-4",
        className
      )}
    >
      <label
        className={clsx(
          "text-[0.92rem] text-[var(--ink-soft)]",
          isCompact && "visually-hidden"
        )}
        htmlFor={inputId}
      >
        Search by title, tags, category, or body text
      </label>
      <input
        id={inputId}
        className="w-full rounded-2xl border border-[var(--line)] bg-white px-[1.1rem] py-4 text-[var(--ink)] outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[color:color-mix(in_srgb,var(--accent-strong)_55%,white)]"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder={
          isCompact
            ? "Search notes in Korean or English"
            : "Search technical notes in Korean or English"
        }
        autoComplete="off"
        autoFocus={autoFocus}
      />
      <div className="text-[0.84rem] text-[var(--ink-soft)]" aria-live="polite">
        {visibleResults.length} results
      </div>
      <ul className={clsx("m-0 grid list-none gap-[0.85rem] p-0", isCompact && "gap-2")}>
        {visibleResults.map((entry) => (
          <li
            key={entry.route}
            className={clsx(
              "border-t border-[var(--line)]",
              isCompact && "first:border-t-0"
            )}
          >
            <Link
              href={entry.route}
              className={clsx(
                "grid gap-1 pt-[0.85rem]",
                isCompact && "gap-[0.15rem] pt-[0.55rem]"
              )}
              onClick={onNavigate}
            >
              <span className="text-[0.84rem] text-[var(--ink-soft)]">
                {new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium" }).format(new Date(entry.date))}
              </span>
              <strong
                className={clsx(
                  "[overflow-wrap:anywhere]",
                  isCompact ? "text-[0.98rem]" : "text-[1.05rem]"
                )}
              >
                {entry.title}
              </strong>
              <span className="leading-[1.6] text-[var(--ink-soft)]">{entry.description}</span>
            </Link>
          </li>
        ))}
      </ul>
      {visibleResults.length === 0 ? (
        <p className="m-0 text-[var(--ink-soft)]">No notes matched that query yet.</p>
      ) : null}
      {isCompact ? (
        <div className="flex justify-end">
          <Link href="/search/" className="text-[0.9rem] text-[var(--accent-strong)]" onClick={onNavigate}>
            Open full search
          </Link>
        </div>
      ) : null}
    </section>
  );
}
