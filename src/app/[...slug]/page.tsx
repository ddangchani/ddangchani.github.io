import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { MotionReveal } from "@/components/motion-reveal";
import { PostComments } from "@/components/post-comments";
import { ReadingProgress } from "@/components/reading-progress";
import { getAllRouteSegments, getPostBySegments } from "@/lib/site-data";

type PostPageProps = {
  params: Promise<{ slug?: string[] }>;
};

export async function generateStaticParams() {
  const routes = await getAllRouteSegments();
  return routes.map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: PostPageProps): Promise<Metadata> {
  const { slug = [] } = await params;
  const post = await getPostBySegments(slug);

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
  const post = await getPostBySegments(slug);

  if (!post) {
    notFound();
  }

  return (
    <div className="page-stack">
      <ReadingProgress />
      <MotionReveal>
        <article className="post-shell">
          <header className="post-shell__header">
            <p className="section-kicker">{post.categories[0] ?? "Archive"}</p>
            <h1 className="page-title">{post.title}</h1>
            <p className="post-shell__summary">{post.description || post.excerpt}</p>
            <div className="post-shell__meta">
              <span>{new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium" }).format(new Date(post.date))}</span>
              <span>{post.readingTimeMinutes} min read</span>
            </div>
          </header>
          {post.teaser ? <img src={post.teaser} alt="" className="post-shell__cover" /> : null}
          {post.headings.length > 0 ? (
            <aside className="post-outline" aria-label="On this page">
              <p className="post-outline__label">On this page</p>
              <ol className="post-outline__list">
                {post.headings.map((heading) => (
                  <li
                    key={heading.id}
                    className="post-outline__item"
                    data-level={heading.level}
                  >
                    <a href={`#${heading.id}`}>{heading.text}</a>
                  </li>
                ))}
              </ol>
            </aside>
          ) : null}
          <section
            className="prose-panel prose-panel--post"
            dangerouslySetInnerHTML={{ __html: post.html }}
          />
          <footer className="post-shell__footer">
            <ul className="post-shell__tags" aria-label="Tags">
              {post.tags.map((tag) => (
                <li key={tag}>{tag}</li>
              ))}
            </ul>
          </footer>
          <PostComments pathname={post.route} />
        </article>
      </MotionReveal>
    </div>
  );
}
