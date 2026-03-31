"use client";

import { useDeferredValue, useState } from "react";

import type { SiteSearchEntry } from "@/lib/site-data";

type SearchPanelProps = {
  entries: SiteSearchEntry[];
};

function matches(entry: SiteSearchEntry, query: string) {
  const haystack = [entry.title, entry.description, entry.bodyPlain, entry.tags.join(" "), entry.categories.join(" ")]
    .join(" ")
    .toLocaleLowerCase();

  return haystack.includes(query.toLocaleLowerCase());
}

export function SearchPanel({ entries }: SearchPanelProps) {
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query.trim());
  const results = deferredQuery
    ? entries.filter((entry) => matches(entry, deferredQuery)).slice(0, 18)
    : entries.slice(0, 12);

  return (
    <section className="search-panel">
      <label className="search-panel__label" htmlFor="search-query">
        Search by title, tags, category, or body text
      </label>
      <input
        id="search-query"
        className="search-panel__input"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search technical notes in Korean or English"
        autoComplete="off"
      />
      <div className="search-panel__count" aria-live="polite">
        {results.length} results
      </div>
      <ul className="search-panel__results">
        {results.map((entry) => (
          <li key={entry.route} className="search-panel__result">
            <a href={entry.route} className="search-panel__result-link">
              <span className="search-panel__result-date">
                {new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium" }).format(new Date(entry.date))}
              </span>
              <strong className="search-panel__result-title">{entry.title}</strong>
              <span className="search-panel__result-description">{entry.description}</span>
            </a>
          </li>
        ))}
      </ul>
      {results.length === 0 ? (
        <p className="search-panel__empty">No notes matched that query yet.</p>
      ) : null}
    </section>
  );
}
