import type { Metadata } from "next";
import Link from "next/link";

import { MotionReveal } from "@/components/motion-reveal";
import { SectionHeading } from "@/components/section-heading";
import { getTagSummaries } from "@/lib/site-data";

export const metadata: Metadata = {
  title: "Tags",
  description: "Browse the archive by tag."
};

export default async function TagsIndexPage() {
  const tags = await getTagSummaries();

  return (
    <div className="page-stack">
      <MotionReveal>
        <section className="content-section">
          <SectionHeading
            eyebrow="Tags"
            title="Browse by topic"
            description="A lightweight tag index for entering the archive by subject instead of chronology."
          />
          <div className="topic-cloud">
            {tags.map((tag) => (
              <Link
                key={tag.name}
                href={`/tags/${encodeURIComponent(tag.name)}/`}
                className="topic-cloud__item"
              >
                <strong>{tag.name}</strong>
                <span>{tag.count} notes</span>
              </Link>
            ))}
          </div>
        </section>
      </MotionReveal>
    </div>
  );
}
