import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { MotionReveal } from "@/components/motion-reveal";
import { PostCard } from "@/components/post-card";
import { SectionHeading } from "@/components/section-heading";
import { decodeRouteSegment, encodeRouteSegment } from "@/lib/content/legacy-routes";
import { getAllTags, getPostsByTag } from "@/lib/site-data";

type TagPageProps = {
  params: Promise<{ tag: string }>;
};

export async function generateStaticParams() {
  const tags = await getAllTags();
  return tags.map((tag) => ({ tag: encodeRouteSegment(tag) }));
}

export async function generateMetadata({ params }: TagPageProps): Promise<Metadata> {
  const { tag: rawTag } = await params;
  const tag = decodeRouteSegment(rawTag);
  return {
    title: `Tag: ${tag}`,
    description: `Posts tagged with ${tag}.`
  };
}

export const dynamicParams = false;

export default async function TagPage({ params }: TagPageProps) {
  const { tag: rawTag } = await params;
  const tag = decodeRouteSegment(rawTag);
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
          <div className="grid grid-cols-1 gap-[1.3rem] md:grid-cols-2">
            {posts.map((post) => (
              <PostCard key={post.route} post={post} compact />
            ))}
          </div>
        </section>
      </MotionReveal>
    </div>
  );
}
