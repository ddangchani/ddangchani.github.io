import Link from "next/link";

export default function NotFound() {
  return (
    <div className="page-stack">
      <section className="content-section content-section--narrow">
        <p className="section-kicker">404</p>
        <h1 className="page-title">This note is not in the exported archive.</h1>
        <p className="prose-panel">
          The route either has not been generated yet or the legacy URL needs to be added to the
          migration manifest.
        </p>
        <Link href="/" className="hero-shell__button hero-shell__button--primary">
          Return home
        </Link>
      </section>
    </div>
  );
}
