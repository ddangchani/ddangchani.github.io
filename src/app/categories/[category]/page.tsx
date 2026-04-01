import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { MotionReveal } from "@/components/motion-reveal";
import { PostCard } from "@/components/post-card";
import { SectionHeading } from "@/components/section-heading";
import { decodeRouteSegment, encodeRouteSegment } from "@/lib/content/legacy-routes";
import { getAllCategories, getPostsByCategory } from "@/lib/site-data";

type CategoryPageProps = {
  params: Promise<{ category: string }>;
};

export async function generateStaticParams() {
  const categories = await getAllCategories();
  return categories.map((category) => ({ category: encodeRouteSegment(category) }));
}

export async function generateMetadata({ params }: CategoryPageProps): Promise<Metadata> {
  const { category: rawCategory } = await params;
  const category = decodeRouteSegment(rawCategory);
  return {
    title: `Category: ${category}`,
    description: `Posts in ${category}.`
  };
}

export const dynamicParams = false;

export default async function CategoryPage({ params }: CategoryPageProps) {
  const { category: rawCategory } = await params;
  const category = decodeRouteSegment(rawCategory);
  const posts = await getPostsByCategory(category);

  if (!posts.length) {
    notFound();
  }

  return (
    <div className="page-stack">
      <MotionReveal>
        <section className="content-section">
          <SectionHeading
            eyebrow="Category"
            title={category}
            description={`Archive entries filed under ${category}.`}
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
