import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { clsx } from "clsx";

import { MotionReveal } from "@/components/motion-reveal";
import { PostComments } from "@/components/post-comments";
import { PostCommentsLoader } from "@/components/post-comments-loader";
import {
  POST_PROSE_END_SENTINEL_ID,
  POST_PROSE_SENTINEL_ID,
  PostStickyTableOfContents,
  PostTableOfContents
} from "@/components/post-table-of-contents";
import { ReadingProgress } from "@/components/reading-progress";
import { TagChip } from "@/components/tag-chip";
import {
  encodeRouteSegment,
  normalizeRouteSegments
} from "@/lib/content/legacy-routes";
import { isFilterTag } from "@/lib/filter-tags";
import { buildTagFilterHref } from "@/lib/tag-query";
import { getAllRouteSegments, getPostBySegments } from "@/lib/site-data";

const postProseClassName =
  "min-w-0 overflow-hidden rounded-[var(--radius-lg)] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_96%,white)] px-[clamp(1.2rem,3vw,2rem)] py-[clamp(1.2rem,3vw,2rem)] leading-[1.85] shadow-[var(--shadow)] max-[720px]:p-4 [&_p]:[overflow-wrap:anywhere] [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_h1]:mt-[2.2rem] [&_h1]:mb-4 [&_h1]:[font-family:var(--font-display),serif] [&_h1]:text-[clamp(2rem,4vw,3rem)] [&_h1]:leading-[1.08] [&_h1]:tracking-[-0.03em] [&_h1]:scroll-mt-24 [&_h1]:[overflow-wrap:anywhere] [&_h2]:mt-8 [&_h2]:mb-[0.85rem] [&_h2]:[font-family:var(--font-display),serif] [&_h2]:text-[clamp(1.6rem,3vw,2.2rem)] [&_h2]:leading-[1.08] [&_h2]:tracking-[-0.03em] [&_h2]:scroll-mt-24 [&_h2]:[overflow-wrap:anywhere] [&_h3]:mt-[1.6rem] [&_h3]:mb-[0.75rem] [&_h3]:[font-family:var(--font-display),serif] [&_h3]:text-[clamp(1.25rem,2.4vw,1.6rem)] [&_h3]:leading-[1.08] [&_h3]:tracking-[-0.03em] [&_h3]:scroll-mt-24 [&_h3]:[overflow-wrap:anywhere] [&_h4]:[font-family:var(--font-display),serif] [&_h4]:leading-[1.08] [&_h4]:tracking-[-0.03em] [&_h4]:scroll-mt-24 [&_h4]:[overflow-wrap:anywhere] [&_ul]:pl-5 [&_ol]:pl-5 [&_li]:[overflow-wrap:anywhere] [&_li+li]:mt-[0.35rem] [&_blockquote]:my-6 [&_blockquote]:border-l-[3px] [&_blockquote]:border-l-[color:color-mix(in_srgb,var(--accent-strong)_45%,var(--line))] [&_blockquote]:py-[0.35rem] [&_blockquote]:pl-4 [&_blockquote]:text-[var(--ink-soft)] [&_blockquote]:[overflow-wrap:anywhere] [&_hr]:my-8 [&_hr]:border-0 [&_hr]:border-t [&_hr]:border-t-[var(--line)] [&_a]:text-[var(--accent-strong)] [&_a]:underline [&_a]:decoration-[0.08em] [&_a]:[overflow-wrap:anywhere] [&_img]:mx-auto [&_img]:my-6 [&_img]:h-auto [&_img]:max-w-full [&_img]:rounded-[1.25rem] [&_pre]:max-w-full [&_pre]:overflow-x-auto [&_pre]:rounded-2xl [&_pre]:bg-[color:color-mix(in_srgb,var(--ink)_6%,white)] [&_pre]:px-[1.1rem] [&_pre]:py-4 [&_pre]:[font-family:var(--font-mono),monospace] [&_code]:[font-family:var(--font-mono),monospace] [&_:not(pre)>code]:rounded-[0.45rem] [&_:not(pre)>code]:bg-[color:color-mix(in_srgb,var(--ink)_6%,white)] [&_:not(pre)>code]:px-[0.35rem] [&_:not(pre)>code]:py-[0.15rem] [&_:not(pre)>code]:[overflow-wrap:anywhere] [&_table]:my-6 [&_table]:block [&_table]:max-w-full [&_table]:overflow-x-auto [&_table]:border-collapse [&_table]:[-webkit-overflow-scrolling:touch] [&_table]:whitespace-nowrap [&_th]:border-b [&_th]:border-b-[var(--line)] [&_th]:px-3 [&_th]:py-3 [&_th]:text-left [&_td]:border-b [&_td]:border-b-[var(--line)] [&_td]:px-3 [&_td]:py-3 [&_td]:text-left [&_.katex-display]:overflow-x-auto [&_.katex-display]:overflow-y-hidden [&_.katex-display]:py-[0.4rem]";

const desktopTocArticleClassName =
  "mx-auto w-full max-w-[900px] min-w-0 min-[1080px]:max-w-[min(1116px,calc(100vw-(var(--page-gutter)*2)))]";

const desktopTocGridClassName =
  "min-[1080px]:grid-cols-[minmax(0,1fr)_12rem] min-[1080px]:items-start";

type PostPageProps = {
  params: Promise<{ slug?: string[] }>;
};

export async function generateStaticParams() {
  const routes = await getAllRouteSegments();
  return routes.map((slug) => ({ slug: slug.map((segment) => encodeRouteSegment(segment)) }));
}

export async function generateMetadata({ params }: PostPageProps): Promise<Metadata> {
  const { slug = [] } = await params;
  const post = await getPostBySegments(normalizeRouteSegments(slug));

  if (!post) {
    return {};
  }

  return {
    title: post.title,
    description: post.description
  };
}

export const dynamicParams = false;

export default async function PostPage({ params }: PostPageProps) {
  const { slug = [] } = await params;
  const post = await getPostBySegments(normalizeRouteSegments(slug));

  if (!post) {
    notFound();
  }

  const hasHeadings = post.headings.length > 0;

  return (
    <div className="page-stack">
      <ReadingProgress />
      <PostCommentsLoader pathname={post.route} />
      <article className={desktopTocArticleClassName}>
        <div
          className={clsx(
            "grid gap-6",
            hasHeadings && desktopTocGridClassName
          )}
        >
          <MotionReveal className="grid min-w-0 gap-6">
            <header className="grid min-w-0 gap-4">
              <p className="section-kicker">{post.categories[0] ?? "Article"}</p>
              <h1 className="page-title">{post.title}</h1>
              <div className="flex min-w-0 flex-wrap gap-x-[0.85rem] gap-y-[0.45rem] text-[0.84rem] text-[var(--ink-soft)] max-[480px]:gap-x-[0.85rem]">
                <span>{new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium" }).format(new Date(post.date))}</span>
                <span>{post.readingTimeMinutes} min read</span>
              </div>
            </header>
            {hasHeadings ? <PostTableOfContents headings={post.headings} /> : null}
            <div className="grid gap-0">
              {hasHeadings ? <div id={POST_PROSE_SENTINEL_ID} aria-hidden="true" className="h-px w-full" /> : null}
              <section
                className={postProseClassName}
                dangerouslySetInnerHTML={{ __html: post.html }}
              />
            </div>
            <footer className="flex min-w-0 flex-wrap items-center justify-between gap-[0.9rem]">
              <ul className="m-0 flex min-w-0 list-none flex-wrap gap-[0.55rem] p-0" aria-label="Tags">
                {post.tags.map((tag) => (
                  <li key={tag} className="min-w-0 max-[480px]:max-w-full max-[480px]:[overflow-wrap:anywhere]">
                    {isFilterTag(tag) ? (
                      <TagChip label={tag} href={buildTagFilterHref(tag)} />
                    ) : (
                      <TagChip label={tag} muted />
                    )}
                  </li>
                ))}
              </ul>
            </footer>
            {hasHeadings ? <div id={POST_PROSE_END_SENTINEL_ID} aria-hidden="true" className="h-px w-full" /> : null}
          </MotionReveal>
          {hasHeadings ? <PostStickyTableOfContents headings={post.headings} /> : null}
        </div>
        <div className="mt-6">
          <PostComments />
        </div>
      </article>
    </div>
  );
}
