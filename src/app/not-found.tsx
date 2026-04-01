import Link from "next/link";

export default function NotFound() {
  return (
    <div className="page-stack">
      <section className="content-section content-section--narrow">
        <p className="section-kicker">404</p>
        <h1 className="page-title">This note is not in the exported archive.</h1>
        <p className="rounded-[var(--radius-lg)] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_90%,white)] px-[clamp(1.2rem,3vw,2rem)] py-[clamp(1.2rem,3vw,2rem)] leading-[1.85] shadow-[var(--shadow)] max-[720px]:p-4">
          The route either has not been generated yet or the legacy URL needs to be added to the
          migration manifest.
        </p>
        <Link
          href="/"
          className="inline-flex items-center justify-center rounded-full border border-transparent bg-[color:color-mix(in_srgb,white_94%,var(--paper)_6%)] px-5 py-3 text-[oklch(0.3_0.08_248)] transition duration-200 ease-out hover:-translate-y-px hover:bg-[color:color-mix(in_srgb,var(--accent)_10%,white)]"
        >
          Return home
        </Link>
      </section>
    </div>
  );
}
