import type { Metadata } from "next";
import Link from "next/link";

import { MotionReveal } from "@/components/motion-reveal";
import { SectionHeading } from "@/components/section-heading";
import { getCategorySummaries } from "@/lib/site-data";

export const metadata: Metadata = {
  title: "Categories",
  description: "Browse the archive by category."
};

export default async function CategoriesIndexPage() {
  const categories = await getCategorySummaries();

  return (
    <div className="page-stack">
      <MotionReveal>
        <section className="content-section">
          <SectionHeading
            eyebrow="Categories"
            title="Browse by collection"
            description="Category landing pages provide a second discovery path alongside tags and search."
          />
          <div className="topic-cloud">
            {categories.map((category) => (
              <Link
                key={category.name}
                href={`/categories/${encodeURIComponent(category.name)}/`}
                className="topic-cloud__item"
              >
                <strong>{category.name}</strong>
                <span>{category.count} notes</span>
              </Link>
            ))}
          </div>
        </section>
      </MotionReveal>
    </div>
  );
}
