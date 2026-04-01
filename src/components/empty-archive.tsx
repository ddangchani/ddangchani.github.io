type EmptyArchiveProps = {
  title: string;
  description: string;
};

export function EmptyArchive({ title, description }: EmptyArchiveProps) {
  return (
    <div
      className="rounded-[var(--radius-lg)] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_90%,white)] px-[clamp(1.2rem,3vw,2rem)] py-[clamp(1.2rem,3vw,2rem)] shadow-[var(--shadow)] max-[720px]:p-4"
      role="status"
    >
      <p className="m-0 text-[0.72rem] uppercase tracking-[0.24em] text-[var(--ink-soft)] max-[480px]:tracking-[0.18em]">
        Waiting on content data
      </p>
      <h2 className="m-0 mt-[0.4rem] [font-family:var(--font-display),serif] text-[2rem] leading-[1.08] tracking-[-0.03em]">
        {title}
      </h2>
      <p className="m-0 mt-3 text-[1.02rem] leading-[1.8] text-[var(--ink-soft)] max-[720px]:text-[0.97rem] max-[720px]:leading-[1.7]">
        {description}
      </p>
    </div>
  );
}
