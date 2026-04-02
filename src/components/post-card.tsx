import Image from "next/image";
import Link from "next/link";
import { clsx } from "clsx";

import { TagChip } from "@/components/tag-chip";
import { isFilterTag } from "@/lib/filter-tags";
import { buildTagFilterHref } from "@/lib/tag-query";

type PostCardProps = {
  post: {
    route: string;
    teaser: string | null;
    date: string;
    categories: string[];
    title: string;
    description: string;
    excerpt: string;
    tags: string[];
  };
  compact?: boolean;
};

export function PostCard({ post, compact = false }: PostCardProps) {
  const route = post.route;
  const visibleTags = post.tags.slice(0, compact ? 2 : 4);
  const primaryCategory = post.categories[0] ?? "Archive";

  return (
    <article className="grid min-w-0 overflow-hidden rounded-[var(--radius-lg)] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_90%,white)] shadow-[var(--shadow)]">
      <div className="relative aspect-[16/9] overflow-hidden bg-[linear-gradient(135deg,color-mix(in_srgb,var(--accent)_22%,white),transparent)]">
        {post.teaser ? (
          <Image src={post.teaser} alt="" fill unoptimized className="h-full w-full object-cover" sizes="(max-width: 960px) 100vw, 50vw" />
        ) : (
          <div
            aria-hidden="true"
            className="absolute inset-0 bg-[radial-gradient(circle_at_18%_20%,color-mix(in_srgb,var(--accent)_18%,white),transparent_28%),radial-gradient(circle_at_82%_24%,color-mix(in_srgb,var(--paper-strong)_48%,white),transparent_24%),linear-gradient(135deg,color-mix(in_srgb,white_72%,var(--paper-strong)_28%),color-mix(in_srgb,white_94%,var(--surface)_6%))]"
          >
            <div className="absolute inset-0 opacity-70 [background-image:linear-gradient(180deg,color-mix(in_srgb,var(--ink)_8%,transparent)_0,color-mix(in_srgb,var(--ink)_8%,transparent)_1px,transparent_1px,transparent_24px)]" />
            <div className="absolute inset-x-5 top-5 flex items-center justify-between gap-3 max-[480px]:inset-x-4 max-[480px]:top-4">
              <span className="inline-flex items-center rounded-full border border-[color:color-mix(in_srgb,var(--accent-strong)_12%,var(--line))] bg-[color:color-mix(in_srgb,white_74%,var(--accent)_26%)] px-[0.8rem] py-[0.38rem] text-[0.68rem] uppercase tracking-[0.16em] text-[var(--accent-strong)] shadow-[inset_0_1px_0_color-mix(in_srgb,white_70%,transparent)]">
                {primaryCategory}
              </span>
              <span className="text-[0.66rem] uppercase tracking-[0.24em] text-[color:color-mix(in_srgb,var(--ink-soft)_82%,var(--paper)_18%)]">
                Editorial note
              </span>
            </div>
            <div className="absolute inset-x-5 bottom-5 grid gap-[0.55rem] max-[480px]:inset-x-4 max-[480px]:bottom-4">
              <span className="h-[1px] w-[32%] rounded-full bg-[color:color-mix(in_srgb,var(--accent-strong)_18%,transparent)]" />
              <span className="h-[1px] w-[78%] rounded-full bg-[color:color-mix(in_srgb,var(--ink-soft)_22%,transparent)]" />
              <span className="h-[1px] w-[58%] rounded-full bg-[color:color-mix(in_srgb,var(--ink-soft)_14%,transparent)]" />
            </div>
            <div className="absolute right-5 bottom-4 flex h-[5.75rem] w-[5.75rem] items-end justify-end rounded-full border border-[color:color-mix(in_srgb,var(--ink)_8%,transparent)] bg-[color:color-mix(in_srgb,white_32%,transparent)] p-4 [box-shadow:inset_0_1px_0_color-mix(in_srgb,white_68%,transparent)] max-[480px]:right-4 max-[480px]:h-[4.8rem] max-[480px]:w-[4.8rem]">
              <span className="[font-family:var(--font-display),serif] text-[2.6rem] leading-none tracking-[-0.08em] text-[color:color-mix(in_srgb,var(--accent-strong)_28%,var(--paper)_72%)] max-[480px]:text-[2.15rem]">
                {primaryCategory.slice(0, 1)}
              </span>
            </div>
          </div>
        )}
      </div>
      <div className="grid min-w-0 gap-4 p-5 max-[720px]:p-4">
        <div className="flex min-w-0 flex-wrap gap-x-[0.85rem] gap-y-[0.45rem] text-[0.84rem] text-[var(--ink-soft)]">
          <span>{new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium" }).format(new Date(post.date))}</span>
          <span>{primaryCategory}</span>
        </div>
        <h3 className="m-0 text-[1.5rem] leading-[1.15] [overflow-wrap:anywhere] max-[720px]:text-[1.25rem]">
          <Link href={route} className="transition-colors duration-200 hover:text-[var(--accent-strong)]">
            {post.title}
          </Link>
        </h3>
        <p className="m-0 [overflow-wrap:anywhere] leading-[1.7] text-[var(--ink-soft)] max-[720px]:text-[0.97rem] max-[720px]:leading-[1.7]">
          {post.description || post.excerpt}
        </p>
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-[0.9rem]">
          <ul className="m-0 flex min-w-0 flex-1 list-none flex-wrap gap-[0.55rem] p-0" aria-label="Tags">
            {visibleTags.map((tag) => (
              <li key={tag} className="min-w-0 max-[480px]:max-w-full">
                {isFilterTag(tag) ? (
                  <TagChip label={tag} href={buildTagFilterHref(tag)} />
                ) : (
                  <TagChip label={tag} muted />
                )}
              </li>
            ))}
          </ul>
          <Link
            href={route}
            className={clsx(
              "inline-flex shrink-0 items-center rounded-full border border-transparent px-3 py-2 text-[var(--accent-strong)] transition duration-200 ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_28%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_10%,white)]",
              compact && "max-[480px]:w-full max-[480px]:justify-center"
            )}
          >
            Read note
          </Link>
        </div>
      </div>
    </article>
  );
}
