import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { MotionReveal } from "@/components/motion-reveal";
import { PostCard } from "@/components/post-card";
import { SectionHeading } from "@/components/section-heading";
import { getAllTags, getPostsByTag } from "@/lib/site-data";

type TagPageProps = {
  params: Promise<{ tag: string }>;
};

export async function generateStaticParams() {
  const tags = await getAllTags();
  return tags.map((tag) => ({ tag }));
}

export async function generateMetadata({ params }: TagPageProps): Promise<Metadata> {
  const { tag } = await params;
  return {
    title: `Tag: ${tag}`,
    description: `Posts tagged with ${tag}.`
  };
}

export const dynamicParams = false;

export default async function TagPage({ params }: TagPageProps) {
  const { tag } = await params;
  const posts = await getPostsByTag(tag);

  if (!posts.length) {
    notFound();
  }

  return (
    <div className="page-stack">
      <MotionReveal>
        <section className="content-section">
          <SectionHeading
            eyebrow="Tag"
            title={tag}
            description={`Posts collected under ${tag}.`}
          />
          <div className="post-grid">
            {posts.map((post) => (
              <PostCard key={post.route} post={post} compact />
            ))}
          </div>
        </section>
      </MotionReveal>
    </div>
  );
}
