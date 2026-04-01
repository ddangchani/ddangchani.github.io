import Image from "next/image";
import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { MotionReveal } from "@/components/motion-reveal";
import { PostComments } from "@/components/post-comments";
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
  "rounded-[var(--radius-lg)] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_96%,white)] px-[clamp(1.2rem,3vw,2rem)] py-[clamp(1.2rem,3vw,2rem)] leading-[1.85] shadow-[var(--shadow)] max-[720px]:p-4 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_h1]:mt-[2.2rem] [&_h1]:mb-4 [&_h1]:[font-family:var(--font-display),serif] [&_h1]:text-[clamp(2rem,4vw,3rem)] [&_h1]:leading-[1.08] [&_h1]:tracking-[-0.03em] [&_h1]:scroll-mt-24 [&_h2]:mt-8 [&_h2]:mb-[0.85rem] [&_h2]:[font-family:var(--font-display),serif] [&_h2]:text-[clamp(1.6rem,3vw,2.2rem)] [&_h2]:leading-[1.08] [&_h2]:tracking-[-0.03em] [&_h2]:scroll-mt-24 [&_h3]:mt-[1.6rem] [&_h3]:mb-[0.75rem] [&_h3]:[font-family:var(--font-display),serif] [&_h3]:text-[clamp(1.25rem,2.4vw,1.6rem)] [&_h3]:leading-[1.08] [&_h3]:tracking-[-0.03em] [&_h3]:scroll-mt-24 [&_h4]:[font-family:var(--font-display),serif] [&_h4]:leading-[1.08] [&_h4]:tracking-[-0.03em] [&_h4]:scroll-mt-24 [&_ul]:pl-5 [&_ol]:pl-5 [&_li+li]:mt-[0.35rem] [&_blockquote]:my-6 [&_blockquote]:border-l-[3px] [&_blockquote]:border-l-[color:color-mix(in_srgb,var(--accent-strong)_45%,var(--line))] [&_blockquote]:py-[0.35rem] [&_blockquote]:pl-4 [&_blockquote]:text-[var(--ink-soft)] [&_hr]:my-8 [&_hr]:border-0 [&_hr]:border-t [&_hr]:border-t-[var(--line)] [&_a]:text-[var(--accent-strong)] [&_a]:underline [&_a]:decoration-[0.08em] [&_img]:mx-auto [&_img]:my-6 [&_img]:rounded-[1.25rem] [&_pre]:overflow-x-auto [&_pre]:rounded-2xl [&_pre]:bg-[color:color-mix(in_srgb,var(--ink)_6%,white)] [&_pre]:px-[1.1rem] [&_pre]:py-4 [&_pre]:[font-family:var(--font-mono),monospace] [&_code]:[font-family:var(--font-mono),monospace] [&_:not(pre)>code]:rounded-[0.45rem] [&_:not(pre)>code]:bg-[color:color-mix(in_srgb,var(--ink)_6%,white)] [&_:not(pre)>code]:px-[0.35rem] [&_:not(pre)>code]:py-[0.15rem] [&_table]:my-6 [&_table]:w-full [&_table]:border-collapse [&_th]:border-b [&_th]:border-b-[var(--line)] [&_th]:px-3 [&_th]:py-3 [&_th]:text-left [&_td]:border-b [&_td]:border-b-[var(--line)] [&_td]:px-3 [&_td]:py-3 [&_td]:text-left [&_.katex-display]:overflow-x-auto [&_.katex-display]:overflow-y-hidden [&_.katex-display]:py-[0.4rem]";

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

  return (
    <div className="page-stack">
      <ReadingProgress />
      <MotionReveal>
        <article className="mx-auto grid w-full max-w-[900px] gap-6">
          <header className="grid gap-4">
            <p className="section-kicker">{post.categories[0] ?? "Archive"}</p>
            <h1 className="page-title">{post.title}</h1>
            <p className="m-0 text-[1.02rem] leading-[1.8] text-[var(--ink-soft)] max-[720px]:text-[0.97rem] max-[720px]:leading-[1.7]">
              {post.description || post.excerpt}
            </p>
            <div className="flex flex-wrap gap-x-[0.85rem] gap-y-[0.45rem] text-[0.84rem] text-[var(--ink-soft)] max-[480px]:gap-x-[0.85rem]">
              <span>{new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium" }).format(new Date(post.date))}</span>
              <span>{post.readingTimeMinutes} min read</span>
            </div>
          </header>
          {post.teaser ? (
            <Image
              src={post.teaser}
              alt=""
              className="rounded-[2rem] border border-[var(--line)] shadow-[var(--shadow)]"
              width={1600}
              height={900}
              unoptimized
            />
          ) : null}
          {post.headings.length > 0 ? (
            <aside
              className="grid min-w-0 gap-3 rounded-[1.5rem] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_92%,white)] px-5 py-4"
              aria-label="On this page"
            >
              <p className="m-0 text-[0.78rem] uppercase tracking-[0.18em] text-[var(--ink-soft)]">
                On this page
              </p>
              <ol className="m-0 grid min-w-0 list-none gap-2 p-0">
                {post.headings.map((heading) => (
                  <li
                    key={heading.id}
                    className={
                      heading.level === 4
                        ? "min-w-0 pl-8 text-[var(--ink-soft)]"
                        : heading.level === 3
                          ? "min-w-0 pl-4 text-[var(--ink-soft)]"
                          : "min-w-0 text-[var(--ink-soft)]"
                    }
                  >
                    <a
                      href={`#${heading.id}`}
                      className="block w-full [overflow-wrap:anywhere] transition-colors duration-200 hover:text-[var(--accent-strong)]"
                    >
                      {heading.text}
                    </a>
                  </li>
                ))}
              </ol>
            </aside>
          ) : null}
          <section
            className={postProseClassName}
            dangerouslySetInnerHTML={{ __html: post.html }}
          />
          <footer className="flex flex-wrap items-center justify-between gap-[0.9rem]">
            <ul className="m-0 flex list-none flex-wrap gap-[0.55rem] p-0" aria-label="Tags">
              {post.tags.map((tag) => (
                <li key={tag} className="max-[480px]:max-w-full max-[480px]:[overflow-wrap:anywhere]">
                  {isFilterTag(tag) ? (
                    <TagChip label={tag} href={buildTagFilterHref(tag)} />
                  ) : (
                    <TagChip label={tag} muted />
                  )}
                </li>
              ))}
            </ul>
          </footer>
          <PostComments pathname={post.route} />
        </article>
      </MotionReveal>
    </div>
  );
}
