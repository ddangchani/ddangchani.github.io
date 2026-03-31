import Link from "next/link";

import { MotionReveal } from "@/components/motion-reveal";
import { siteConfig } from "@/lib/site-config";

type HeroProps = {
  postCount: number;
  tagCount: number;
};

export function Hero({ postCount, tagCount }: HeroProps) {
  return (
    <section className="hero-shell">
      <MotionReveal className="hero-shell__content">
        <div className="hero-shell__lede">
          <p className="hero-shell__eyebrow">Research archive</p>
          <h1 className="hero-shell__title">
            Technical writing with the cadence of a field notebook, not a template.
          </h1>
          <p className="hero-shell__description">
            Long-form posts on AI, statistics, causal inference, and implementation details,
            organized for rereading instead of endless scrolling.
          </p>
          <div className="hero-shell__actions">
            <Link href="/posts/" className="hero-shell__button hero-shell__button--primary">
              Enter the archive
            </Link>
            <Link href="/search/" className="hero-shell__button hero-shell__button--secondary">
              Search notes
            </Link>
          </div>
        </div>
        <aside className="hero-shell__panel" aria-label="Site highlights">
          <div className="hero-shell__panel-grid">
            <div className="hero-shell__metric">
              <span className="hero-shell__metric-label">Posts</span>
              <strong className="hero-shell__metric-value">{postCount}</strong>
            </div>
            <div className="hero-shell__metric">
              <span className="hero-shell__metric-label">Topics</span>
              <strong className="hero-shell__metric-value">{tagCount}</strong>
            </div>
          </div>
          <div className="hero-shell__highlights">
            {siteConfig.highlights.map((highlight) => (
              <div key={highlight.label} className="hero-shell__highlight">
                <span>{highlight.label}</span>
                <strong>{highlight.value}</strong>
              </div>
            ))}
          </div>
        </aside>
      </MotionReveal>
    </section>
  );
}
