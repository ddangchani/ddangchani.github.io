import Link from "next/link";

import type { SitePost } from "@/lib/site-data";

type PostCardProps = {
  post: SitePost;
  compact?: boolean;
};

export function PostCard({ post, compact = false }: PostCardProps) {
  const route = post.route;

  return (
    <article className={compact ? "post-card post-card--compact" : "post-card"}>
      {post.teaser ? (
        <div className="post-card__visual">
          <img src={post.teaser} alt="" className="post-card__image" />
        </div>
      ) : null}
      <div className="post-card__body">
        <div className="post-card__meta">
          <span>{new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium" }).format(new Date(post.date))}</span>
          <span>{post.categories[0] ?? "Archive"}</span>
        </div>
        <h3 className="post-card__title">
          <Link href={route}>{post.title}</Link>
        </h3>
        <p className="post-card__excerpt">{post.description || post.excerpt}</p>
        <div className="post-card__footer">
          <ul className="post-card__tags" aria-label="Tags">
            {post.tags.slice(0, compact ? 2 : 4).map((tag) => (
              <li key={tag}>{tag}</li>
            ))}
          </ul>
          <Link href={route} className="post-card__readmore">
            Read note
          </Link>
        </div>
      </div>
    </article>
  );
}
