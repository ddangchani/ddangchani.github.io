import type { Metadata } from "next";

import { MotionReveal } from "@/components/motion-reveal";
import { SearchPanel } from "@/components/search-panel";
import { getSearchEntries } from "@/lib/site-data";

export const metadata: Metadata = {
  title: "Search",
  description: "Search technical notes across the archive."
};

export default async function SearchPage() {
  const entries = await getSearchEntries();

  return (
    <div className="page-stack">
      <MotionReveal>
        <section className="content-section content-section--narrow">
          <p className="section-kicker">Search</p>
          <h1 className="page-title">Find a note by concept, title, or implementation detail.</h1>
          <SearchPanel entries={entries} />
        </section>
      </MotionReveal>
    </div>
  );
}
