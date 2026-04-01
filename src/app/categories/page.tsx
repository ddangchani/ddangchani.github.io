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
          <div className="grid gap-[0.9rem] min-[481px]:grid-cols-2 min-[961px]:grid-cols-1">
            {categories.map((category) => (
              <Link
                key={category.name}
                href={`/categories/${category.name}/`}
                className="grid gap-[0.35rem] rounded-[1.4rem] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_88%,white)] px-[1.15rem] py-4 transition duration-200 ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_32%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_10%,white)] max-[720px]:px-4 max-[720px]:py-[0.95rem]"
              >
                <strong>{category.name}</strong>
                <span className="text-[0.86rem] text-[var(--ink-soft)]">{category.count} notes</span>
              </Link>
            ))}
          </div>
        </section>
      </MotionReveal>
    </div>
  );
}
