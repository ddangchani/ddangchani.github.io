import { clsx } from "clsx";
import type { ReactNode } from "react";

type SectionHeadingProps = {
  eyebrow: string;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
};

export function SectionHeading({
  eyebrow,
  title,
  description,
  action,
  className
}: SectionHeadingProps) {
  return (
    <div
      className={clsx(
        "flex min-w-0 flex-col items-start justify-between gap-4 md:flex-row md:items-center",
        className
      )}
    >
      <div className="min-w-0">
        <p className="m-0 text-[0.72rem] uppercase tracking-[0.24em] text-[var(--ink-soft)] max-[480px]:tracking-[0.18em]">
          {eyebrow}
        </p>
        <h2 className="mb-[5px] mt-[0.35rem] [font-family:var(--font-display),serif] text-[clamp(2rem,3vw,3.1rem)] leading-[1.05] tracking-[-0.04em] [overflow-wrap:anywhere] max-[720px]:text-[clamp(1.7rem,8vw,2.3rem)]">
          {title}
        </h2>
        {description ? (
          <p className="m-0 text-[1.02rem] leading-[1.8] text-[var(--ink-soft)] max-[720px]:text-[0.97rem] max-[720px]:leading-[1.7]">
            {description}
          </p>
        ) : null}
      </div>
      {action ? (
        <div className="w-full md:w-auto [&>*]:w-full md:[&>*]:w-auto">{action}</div>
      ) : null}
    </div>
  );
}
