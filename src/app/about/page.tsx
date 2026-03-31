import type { Metadata } from "next";

import { MotionReveal } from "@/components/motion-reveal";
import { siteConfig } from "@/lib/site-config";

export const metadata: Metadata = {
  title: "About",
  description: "About the author and the purpose of the archive."
};

export default function AboutPage() {
  return (
    <div className="page-stack">
      <MotionReveal>
        <section className="content-section content-section--narrow">
          <p className="section-kicker">About</p>
          <h1 className="page-title">{siteConfig.author.name}</h1>
          <div className="prose-panel">
            <p>{siteConfig.author.bio}</p>
            <p>
              This archive focuses on careful technical writing across AI, statistics, and data
              work. The goal is not just to publish, but to make notes worth revisiting.
            </p>
            <p>
              Based in {siteConfig.author.location}. You can reach out via{" "}
              <a href={`mailto:${siteConfig.author.email}`}>{siteConfig.author.email}</a>.
            </p>
          </div>
        </section>
      </MotionReveal>
    </div>
  );
}
