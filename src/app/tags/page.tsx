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
          <div className="grid gap-[0.9rem] min-[481px]:grid-cols-2 min-[961px]:grid-cols-1">
            {tags.map((tag) => (
              <Link
                key={tag.name}
                href={`/tags/${tag.name}/`}
                className="grid gap-[0.35rem] rounded-[1.4rem] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_88%,white)] px-[1.15rem] py-4 transition duration-200 ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_32%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_10%,white)] max-[720px]:px-4 max-[720px]:py-[0.95rem]"
              >
                <strong>{tag.name}</strong>
                <span className="text-[0.86rem] text-[var(--ink-soft)]">{tag.count} notes</span>
              </Link>
            ))}
          </div>
        </section>
      </MotionReveal>
    </div>
  );
}
