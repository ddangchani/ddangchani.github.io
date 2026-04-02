"use client";

import { clsx } from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { SearchPanel } from "@/components/search-panel";
import { siteConfig } from "@/lib/site-config";
import type { SiteSearchEntry } from "@/lib/site-data";

type SiteHeaderProps = {
  entries: SiteSearchEntry[];
};

export function SiteHeader({ entries }: SiteHeaderProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const headerRef = useRef<HTMLElement | null>(null);
  const pathname = usePathname();
  const normalizedPathname = pathname.endsWith("/") ? pathname : `${pathname}/`;

  useEffect(() => {
    if (!isMenuOpen && !isSearchOpen) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsMenuOpen(false);
        setIsSearchOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isMenuOpen, isSearchOpen]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      setIsMenuOpen(false);
      setIsSearchOpen(false);
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [pathname]);

  useEffect(() => {
    if (!isSearchOpen && !isMenuOpen) {
      return undefined;
    }

    const handleDocumentClick = (event: MouseEvent) => {
      const eventPath = event.composedPath();
      const isInsideHeader = headerRef.current ? eventPath.includes(headerRef.current) : false;

      if (!isInsideHeader) {
        setIsMenuOpen(false);
        setIsSearchOpen(false);
      }
    };

    window.addEventListener("click", handleDocumentClick);

    return () => {
      window.removeEventListener("click", handleDocumentClick);
    };
  }, [isMenuOpen, isSearchOpen]);

  function closePanels() {
    setIsMenuOpen(false);
    setIsSearchOpen(false);
  }

  function isNavItemActive(href: string) {
    return normalizedPathname === href || normalizedPathname.startsWith(href);
  }

  return (
    <header
      className="sticky top-0 z-30 border-b border-[var(--line)] bg-[color:color-mix(in_srgb,var(--paper)_88%,transparent)] backdrop-blur-[18px]"
      ref={headerRef}
    >
      <div className="mx-auto grid w-[var(--content-width)] gap-3 pt-[calc(env(safe-area-inset-top)*0.35+0.85rem)] pb-[0.85rem]">
        <div className="flex items-center justify-between gap-4 max-[720px]:flex-wrap">
          <Link
            href="/"
            className="flex min-w-0 flex-col gap-[0.15rem] pr-3"
            aria-label={siteConfig.title}
            onClick={closePanels}
          >
            <span className="m-0 text-[0.72rem] uppercase tracking-[0.24em] text-[var(--ink-soft)] max-[480px]:tracking-[0.18em]">
              Technical Notebook
            </span>
            <span className="[font-family:var(--font-display),serif] text-[1.2rem] [overflow-wrap:anywhere]">
              {siteConfig.title}
            </span>
          </Link>
          <div className="flex items-center justify-end gap-2 max-[720px]:ml-auto max-[720px]:flex-1 max-[720px]:flex-wrap">
            <div className="hidden items-center gap-1 rounded-full border border-[color:color-mix(in_srgb,var(--accent-strong)_16%,var(--line))] bg-[linear-gradient(180deg,color-mix(in_srgb,white_86%,var(--paper-strong)_14%),color-mix(in_srgb,var(--surface)_96%,white_4%))] p-1 shadow-[inset_0_1px_0_color-mix(in_srgb,white_82%,transparent),0_14px_32px_color-mix(in_srgb,var(--accent-strong)_8%,transparent)] min-[721px]:flex">
              <button
                type="button"
                className={clsx(
                  "inline-flex h-[2.75rem] items-center justify-center gap-[0.55rem] rounded-full border px-[0.9rem] text-[0.92rem] font-medium transition duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:color-mix(in_srgb,var(--accent)_18%,transparent)]",
                  isSearchOpen
                    ? "border-[color:color-mix(in_srgb,var(--accent-strong)_28%,var(--line))] bg-[linear-gradient(180deg,color-mix(in_srgb,white_88%,var(--accent)_12%),color-mix(in_srgb,var(--accent)_12%,white))] text-[var(--accent-strong)] shadow-[inset_0_1px_0_color-mix(in_srgb,white_78%,transparent)]"
                    : "border-transparent bg-transparent text-[color:color-mix(in_srgb,var(--ink-soft)_84%,var(--ink)_16%)] hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_18%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_8%,white)] hover:text-[var(--ink)]"
                )}
                aria-expanded={isSearchOpen}
                aria-controls="site-search-panel"
                aria-label={isSearchOpen ? "Close search panel" : "Open search panel"}
                onClick={() => {
                  setIsSearchOpen((current) => !current);
                  setIsMenuOpen(false);
                }}
              >
                <svg
                  className="h-[1rem] w-[1rem]"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <circle cx="11" cy="11" r="7" />
                  <path d="m20 20-3.5-3.5" />
                </svg>
                <span className="hidden min-[860px]:inline">Search</span>
              </button>
              <span
                className="h-6 w-px rounded-full bg-[color:color-mix(in_srgb,var(--accent-strong)_12%,var(--line))]"
                aria-hidden="true"
              />
              <nav className="flex items-center gap-[0.2rem]" aria-label="Primary">
                {siteConfig.nav.map((item) => {
                  const isActive = isNavItemActive(item.href);

                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={clsx(
                        "inline-flex items-center justify-center rounded-full border px-[1rem] py-[0.7rem] text-[0.92rem] font-medium tracking-[0.01em] transition duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:color-mix(in_srgb,var(--accent)_18%,transparent)]",
                        isActive
                          ? "border-[color:color-mix(in_srgb,var(--accent-strong)_26%,var(--line))] bg-[linear-gradient(180deg,color-mix(in_srgb,white_88%,var(--accent)_14%),color-mix(in_srgb,var(--accent)_12%,white))] text-[var(--accent-strong)] shadow-[inset_0_1px_0_color-mix(in_srgb,white_80%,transparent)]"
                          : "border-transparent text-[color:color-mix(in_srgb,var(--ink-soft)_88%,var(--ink)_12%)] hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_14%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--paper-strong)_58%,white_42%)] hover:text-[var(--ink)]"
                      )}
                      onClick={closePanels}
                    >
                      {item.label}
                    </Link>
                  );
                })}
              </nav>
            </div>
            <div className="relative z-10 flex shrink-0 items-center gap-2 min-[721px]:hidden">
              <button
                type="button"
                className={clsx(
                  "inline-flex h-[2.75rem] w-[2.75rem] items-center justify-center rounded-full border bg-[linear-gradient(180deg,color-mix(in_srgb,white_88%,var(--paper-strong)_12%),color-mix(in_srgb,var(--surface)_95%,white_5%))] text-[var(--ink)] shadow-[inset_0_1px_0_color-mix(in_srgb,white_82%,transparent),0_10px_24px_color-mix(in_srgb,var(--accent-strong)_7%,transparent)] transition duration-200 ease-out max-[480px]:h-[2.6rem] max-[480px]:w-[2.6rem]",
                  isSearchOpen
                    ? "border-[color:color-mix(in_srgb,var(--accent-strong)_26%,var(--line))] text-[var(--accent-strong)]"
                    : "border-[color:color-mix(in_srgb,var(--accent-strong)_16%,var(--line))] hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_22%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_8%,white)]"
                )}
                aria-expanded={isSearchOpen}
                aria-controls="site-search-panel"
                aria-label={isSearchOpen ? "Close search panel" : "Open search panel"}
                onClick={() => {
                  setIsSearchOpen((current) => !current);
                  setIsMenuOpen(false);
                }}
              >
                <svg
                  className="h-[1rem] w-[1rem]"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <circle cx="11" cy="11" r="7" />
                  <path d="m20 20-3.5-3.5" />
                </svg>
              </button>
              <button
                type="button"
                className={clsx(
                  "inline-flex shrink-0 items-center gap-3 rounded-full border bg-[linear-gradient(180deg,color-mix(in_srgb,white_88%,var(--paper-strong)_12%),color-mix(in_srgb,var(--surface)_95%,white_5%))] px-[0.95rem] py-[0.78rem] text-[var(--ink)] shadow-[inset_0_1px_0_color-mix(in_srgb,white_82%,transparent),0_10px_24px_color-mix(in_srgb,var(--accent-strong)_7%,transparent)] transition duration-200 ease-out max-[480px]:px-[0.85rem]",
                  isMenuOpen
                    ? "border-[color:color-mix(in_srgb,var(--accent-strong)_26%,var(--line))] text-[var(--accent-strong)]"
                    : "border-[color:color-mix(in_srgb,var(--accent-strong)_16%,var(--line))] hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_22%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_8%,white)]"
                )}
                aria-expanded={isMenuOpen}
                aria-controls="site-navigation"
                aria-label={isMenuOpen ? "Close navigation menu" : "Open navigation menu"}
                onClick={() => {
                  setIsMenuOpen((current) => !current);
                  setIsSearchOpen(false);
                }}
              >
                <span className="text-[0.92rem] text-[var(--ink-soft)] max-[480px]:hidden">Menu</span>
                <span className="grid gap-[0.22rem]" aria-hidden="true">
                  <span className="block h-[1.5px] w-4 rounded-full bg-current" />
                  <span className="block h-[1.5px] w-4 rounded-full bg-current" />
                  <span className="block h-[1.5px] w-4 rounded-full bg-current" />
                </span>
              </button>
            </div>
          </div>
        </div>
        <nav
          id="site-navigation"
          className={clsx(
            "hidden w-full gap-[0.7rem] min-[721px]:hidden",
            isMenuOpen && "grid grid-cols-2 max-[480px]:grid-cols-1"
          )}
          aria-label="Primary"
        >
          {siteConfig.nav.map((item) => {
            const isActive = isNavItemActive(item.href);

            return (
              <Link
                key={item.href}
                href={item.href}
                className={clsx(
                  "inline-flex w-full items-center justify-center rounded-[1.1rem] border px-[1rem] py-[0.85rem] text-[0.94rem] font-medium transition duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:color-mix(in_srgb,var(--accent)_18%,transparent)]",
                  isActive
                    ? "border-[color:color-mix(in_srgb,var(--accent-strong)_26%,var(--line))] bg-[linear-gradient(180deg,color-mix(in_srgb,white_88%,var(--accent)_14%),color-mix(in_srgb,var(--accent)_12%,white))] text-[var(--accent-strong)] shadow-[0_14px_30px_color-mix(in_srgb,var(--accent-strong)_10%,transparent)]"
                    : "border-[color:color-mix(in_srgb,var(--accent-strong)_14%,var(--line))] bg-[linear-gradient(180deg,color-mix(in_srgb,white_88%,var(--paper-strong)_12%),color-mix(in_srgb,var(--surface)_95%,white_5%))] text-[var(--ink)] shadow-[inset_0_1px_0_color-mix(in_srgb,white_82%,transparent),0_12px_26px_color-mix(in_srgb,var(--accent-strong)_7%,transparent)] hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_22%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_8%,white)]"
                )}
                onClick={closePanels}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        {isSearchOpen ? (
          <div id="site-search-panel" className="w-full pt-1">
            <SearchPanel
              entries={entries}
              variant="compact"
              autoFocus
              inputId="site-search-query"
              onNavigate={closePanels}
            />
          </div>
        ) : null}
      </div>
    </header>
  );
}
