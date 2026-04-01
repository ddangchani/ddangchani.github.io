"use client";

import { clsx } from "clsx";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useEffectEvent, useState, type FocusEvent } from "react";

import type { SitePopularEntry } from "@/lib/site-data";

type PopularPostsCarouselProps = {
  entries: SitePopularEntry[];
};

function formatPopularDate(value?: string) {
  if (!value) {
    return null;
  }

  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric"
  }).format(new Date(value));
}

function formatViewCount(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function wrapIndex(index: number, total: number) {
  if (total === 0) {
    return 0;
  }

  return (index + total) % total;
}

export function PopularPostsCarousel({ entries }: PopularPostsCarouselProps) {
  const reducedMotion = useReducedMotion();
  const [activeIndex, setActiveIndex] = useState(0);
  const [direction, setDirection] = useState(1);
  const [hasFocusWithin, setHasFocusWithin] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const total = entries.length;
  const activeEntry = entries[activeIndex];
  const isPaused = hasFocusWithin || isHovered;

  const moveSlide = (nextIndex: number, nextDirection: number) => {
    if (total < 2) {
      return;
    }

    setDirection(nextDirection);
    setActiveIndex(wrapIndex(nextIndex, total));
  };

  const queueNextSlide = useEffectEvent(() => {
    moveSlide(activeIndex + 1, 1);
  });

  useEffect(() => {
    if (total < 2 || reducedMotion || isPaused) {
      return;
    }

    const intervalId = window.setInterval(() => {
      queueNextSlide();
    }, 6200);

    return () => window.clearInterval(intervalId);
  }, [isPaused, reducedMotion, total]);

  if (!activeEntry) {
    return null;
  }

  const activeEntryDate = formatPopularDate(activeEntry.date);

  function handleNavigation(index: number) {
    if (index === activeIndex) {
      return;
    }

    moveSlide(index, index > activeIndex ? 1 : -1);
  }

  function handleFocusExit(event: FocusEvent<HTMLDivElement>) {
    if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) {
      return;
    }

    setHasFocusWithin(false);
  }

  return (
    <div
      className="grid gap-4 min-[721px]:items-stretch min-[721px]:grid-cols-[minmax(0,1.45fr)_minmax(18rem,0.95fr)]"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onFocusCapture={() => setHasFocusWithin(true)}
      onBlurCapture={handleFocusExit}
    >
      <div className="grid h-full gap-4 overflow-hidden rounded-[var(--radius-lg)] border border-[var(--line)] bg-[radial-gradient(circle_at_100%_0%,color-mix(in_srgb,var(--accent)_12%,transparent),transparent_30%),linear-gradient(180deg,color-mix(in_srgb,white_80%,var(--paper-strong)_20%),white)] p-[clamp(1rem,2vw,1.3rem)] shadow-[var(--shadow)]">
        <div className="flex flex-col gap-4 min-[721px]:flex-row min-[721px]:items-center min-[721px]:justify-between">
          <div className="grid gap-[0.3rem]">
            <span className="text-[0.72rem] uppercase tracking-[0.22em] text-[var(--ink-soft)]">
              Reader picks
            </span>
            <span className="[font-family:var(--font-display),serif] text-base tracking-[-0.04em]" aria-live="polite">
              {String(activeIndex + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}
            </span>
          </div>
          {total > 1 ? (
            <div className="inline-flex w-full items-center gap-2 min-[481px]:w-auto" aria-label="Popular post controls">
              <button
                type="button"
                className="flex-1 rounded-full border border-[var(--line)] bg-[color:color-mix(in_srgb,white_88%,var(--surface)_12%)] px-[0.95rem] py-[0.65rem] text-[0.88rem] transition duration-200 ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_28%,var(--line))] hover:bg-[color:color-mix(in_srgb,white_75%,var(--accent)_25%)] min-[481px]:flex-none"
                onClick={() => moveSlide(activeIndex - 1, -1)}
                aria-label="Previous popular post"
              >
                Prev
              </button>
              <button
                type="button"
                className="flex-1 rounded-full border border-[var(--line)] bg-[color:color-mix(in_srgb,white_88%,var(--surface)_12%)] px-[0.95rem] py-[0.65rem] text-[0.88rem] transition duration-200 ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_28%,var(--line))] hover:bg-[color:color-mix(in_srgb,white_75%,var(--accent)_25%)] min-[481px]:flex-none"
                onClick={() => moveSlide(activeIndex + 1, 1)}
                aria-label="Next popular post"
              >
                Next
              </button>
            </div>
          ) : null}
        </div>

        <div className="relative min-h-[clamp(23rem,42vw,31rem)]" aria-live="polite">
          <AnimatePresence initial={false} custom={direction} mode="wait">
            <motion.article
              key={activeEntry.route}
              className="absolute inset-0"
              custom={direction}
              initial={reducedMotion ? { opacity: 0 } : { opacity: 0, x: direction > 0 ? 32 : -32 }}
              animate={{ opacity: 1, x: 0 }}
              exit={reducedMotion ? { opacity: 0 } : { opacity: 0, x: direction > 0 ? -28 : 28 }}
              transition={{ duration: reducedMotion ? 0.16 : 0.42, ease: [0.16, 1, 0.3, 1] }}
            >
              <Link
                href={activeEntry.route}
                className="group grid h-full grid-rows-[minmax(10.5rem,13rem)_minmax(0,1fr)] overflow-hidden rounded-[calc(var(--radius-lg)-0.35rem)] border border-[color:color-mix(in_srgb,var(--accent)_10%,var(--line))] bg-[color:color-mix(in_srgb,var(--surface)_94%,white)] transition duration-200 ease-out hover:-translate-y-[2px] hover:border-[color:color-mix(in_srgb,var(--accent-strong)_25%,var(--line))] hover:shadow-[0_24px_48px_color-mix(in_srgb,var(--accent-strong)_10%,transparent)] max-[720px]:grid-rows-[minmax(9.5rem,11.5rem)_minmax(0,1fr)]"
              >
                <div className="relative min-h-0 overflow-hidden bg-[radial-gradient(circle_at_30%_30%,color-mix(in_srgb,var(--accent)_18%,transparent),transparent_32%),linear-gradient(135deg,color-mix(in_srgb,var(--paper-strong)_88%,white),white)]">
                  {activeEntry.teaser ? (
                    <Image
                      src={activeEntry.teaser}
                      alt=""
                      fill
                      sizes="(max-width: 720px) 100vw, (max-width: 1200px) 60vw, 42vw"
                      unoptimized
                      className="h-full w-full object-cover transition duration-300 ease-out group-hover:scale-[1.035] group-hover:[filter:saturate(1.05)]"
                    />
                  ) : (
                    <div className="h-full w-full" aria-hidden="true" />
                  )}
                </div>
                <div className="grid content-start gap-3 p-[clamp(1.2rem,3vw,1.7rem)]">
                  <div className="flex flex-wrap gap-[0.7rem] text-[0.82rem] text-[var(--ink-soft)] max-[480px]:gap-x-[0.85rem] max-[480px]:gap-y-[0.45rem]">
                    {activeEntryDate ? <span>{activeEntryDate}</span> : null}
                    {activeEntry.categories?.[0] ? <span>{activeEntry.categories[0]}</span> : null}
                    <span>{formatViewCount(activeEntry.viewCount)} views</span>
                  </div>
                  <h3 className="m-0 [font-family:var(--font-display),serif] text-[clamp(1.72rem,3vw,2.3rem)] leading-[1.08] tracking-[-0.03em]">
                    {activeEntry.title ?? activeEntry.route}
                  </h3>
                  <p className="m-0 overflow-hidden [display:-webkit-box] [-webkit-box-orient:vertical] [-webkit-line-clamp:4] leading-[1.75] text-[var(--ink-soft)] max-[720px]:text-[0.97rem] max-[720px]:leading-[1.7]">
                    {activeEntry.excerpt}
                  </p>
                  <span className="mt-[0.2rem] inline-flex items-center gap-[0.65rem] justify-self-start text-[0.86rem] font-semibold uppercase tracking-[0.08em] text-[var(--accent-strong)] before:h-px before:w-[2.3rem] before:bg-[color:color-mix(in_srgb,var(--accent-strong)_48%,transparent)] before:transition-all before:duration-200 before:ease-out group-hover:before:w-[3.3rem] group-hover:before:bg-[var(--accent-strong)]">
                    Read article
                  </span>
                </div>
              </Link>
            </motion.article>
          </AnimatePresence>
        </div>

        <div className="h-[3px] overflow-hidden rounded-full bg-[color:color-mix(in_srgb,var(--ink)_8%,transparent)]" aria-hidden="true">
          <span
            className="block h-full w-full origin-left bg-[linear-gradient(90deg,color-mix(in_srgb,var(--accent)_70%,white),var(--accent-strong))] transition-transform duration-300 ease-out"
            style={{ transform: `scaleX(${(activeIndex + 1) / total})` }}
          />
        </div>
      </div>

      {total > 1 ? (
        <div
          className="grid h-full content-start gap-[0.35rem] overflow-hidden rounded-[var(--radius-lg)] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_92%,white)] p-[0.45rem] shadow-[var(--shadow)] max-[720px]:grid-flow-col max-[720px]:auto-cols-[minmax(16rem,18rem)] max-[720px]:overflow-x-auto max-[720px]:pb-[0.35rem]"
          aria-label="Popular post list"
        >
          {entries.map((entry, index) => {
            const isActive = index === activeIndex;
            const entryDate = formatPopularDate(entry.date);

            return (
              <button
                key={entry.route}
                type="button"
                className={clsx(
                  "relative grid min-h-full grid-cols-[auto_minmax(0,1fr)] items-start gap-[0.9rem] rounded-[1.35rem] border border-transparent bg-transparent px-4 py-[0.95rem] text-left transition duration-200 ease-out before:absolute before:top-[0.9rem] before:bottom-[0.9rem] before:left-[0.95rem] before:w-[3px] before:origin-center before:scale-y-[0.35] before:rounded-full before:bg-[color:color-mix(in_srgb,var(--accent-strong)_18%,transparent)] before:opacity-0 before:transition-all before:duration-200 before:ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_22%,var(--line))] hover:bg-[color:color-mix(in_srgb,white_84%,var(--paper-strong)_16%)] focus-visible:border-[color:color-mix(in_srgb,var(--accent)_22%,var(--line))] focus-visible:bg-[color:color-mix(in_srgb,white_84%,var(--paper-strong)_16%)] focus-visible:outline-none max-[720px]:snap-start",
                  isActive &&
                    "border-[color:color-mix(in_srgb,var(--accent-strong)_24%,var(--line))] bg-[color:color-mix(in_srgb,white_76%,var(--accent)_24%)] shadow-[inset_0_0_0_1px_color-mix(in_srgb,white_55%,transparent)] before:scale-y-100 before:bg-[linear-gradient(180deg,color-mix(in_srgb,var(--accent)_68%,white),var(--accent-strong))] before:opacity-100"
                )}
                onClick={() => handleNavigation(index)}
                aria-pressed={isActive}
                aria-label={`Show popular post ${index + 1}: ${entry.title ?? entry.route}`}
              >
                <span className="inline-flex min-w-8 justify-center [font-family:var(--font-display),serif] text-[0.96rem] tracking-[-0.04em] text-[var(--ink-soft)]">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <span className="grid gap-[0.45rem]">
                  <span className="flex flex-wrap gap-[0.7rem] text-[0.82rem] text-[var(--ink-soft)]">
                    {entryDate ? <span>{entryDate}</span> : null}
                    <span>{formatViewCount(entry.viewCount)} views</span>
                  </span>
                  <strong className="m-0 [font-family:var(--font-display),serif] text-[1.1rem] leading-[1.28] tracking-[-0.03em]">
                    {entry.title ?? entry.route}
                  </strong>
                </span>
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
