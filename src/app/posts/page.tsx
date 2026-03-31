import type { Metadata } from "next";

import { EmptyArchive } from "@/components/empty-archive";
import { MotionReveal } from "@/components/motion-reveal";
import { PostCard } from "@/components/post-card";
import { SectionHeading } from "@/components/section-heading";
import { getAllPosts } from "@/lib/site-data";

export const metadata: Metadata = {
  title: "Archive",
  description: "Browse every published note in the archive."
};

export default async function PostsPage() {
  const posts = await getAllPosts();

  return (
    <div className="page-stack">
      <MotionReveal>
        <section className="content-section">
          <SectionHeading
            eyebrow="Archive"
            title="All posts"
            description="A chronological reading room for the full body of work."
          />
          {posts.length > 0 ? (
            <div className="post-grid">
              {posts.map((post) => (
                <PostCard key={post.route} post={post} compact />
              ))}
            </div>
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
