import type { Metadata } from "next";
import { Suspense } from "react";

import { EmptyArchive } from "@/components/empty-archive";
import { MotionReveal } from "@/components/motion-reveal";
import { PostCard } from "@/components/post-card";
import { PostsArchive } from "@/components/posts-archive";
import { getAllPosts, getFilterTagSummaries } from "@/lib/site-data";

export const metadata: Metadata = {
  title: "Archive",
  description: "Browse every published note in the archive."
};

export default async function PostsPage() {
  const [posts, tags] = await Promise.all([getAllPosts(), getFilterTagSummaries()]);
  const archivePosts = posts.map((post) => ({
    route: post.route,
    title: post.title,
    description: post.description,
    excerpt: post.excerpt,
    date: post.date,
    categories: post.categories,
    tags: post.tags,
    teaser: post.teaser
  }));

  return (
    <div className="page-stack">
      <MotionReveal>
        <section className="content-section">
          {posts.length > 0 ? (
            <Suspense
              fallback={
                <div className="grid grid-cols-1 gap-[1.3rem] md:grid-cols-2">
                  {archivePosts.map((post) => (
                    <PostCard key={post.route} post={post} compact />
                  ))}
                </div>
              }
            >
              <PostsArchive posts={archivePosts} tags={tags} />
            </Suspense>
          ) : (
            <EmptyArchive
              title="The archive is ready for posts."
              description="This page will hydrate from the generated post manifest once the content migration completes."
            />
          )}
        </section>
      </MotionReveal>
    </div>
  );
}
