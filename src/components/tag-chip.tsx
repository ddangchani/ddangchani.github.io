import Link from "next/link";
import { clsx } from "clsx";

const tagChipBaseClassName =
  "relative inline-flex min-h-8 items-center justify-center overflow-hidden rounded-full px-[0.8rem] py-[0.4rem] pl-[1.2rem] text-[0.82rem] before:absolute before:left-[0.62rem] before:top-1/2 before:h-[0.38rem] before:w-[0.38rem] before:-translate-y-1/2 before:rounded-full";

const tagChipFilterClassName =
  "border border-[var(--tag-filter-line)] bg-[linear-gradient(180deg,color-mix(in_srgb,white_94%,var(--accent)_6%),color-mix(in_srgb,var(--tag-filter-fill)_88%,white_12%))] text-[color:color-mix(in_srgb,var(--accent-strong)_88%,var(--ink)_12%)] shadow-[inset_0_1px_0_color-mix(in_srgb,white_75%,transparent),0_10px_22px_color-mix(in_srgb,var(--accent-strong)_10%,transparent)] before:bg-[color:color-mix(in_srgb,var(--accent)_68%,white)] before:shadow-[0_0_0_0.22rem_color-mix(in_srgb,var(--accent)_10%,transparent)]";

const tagChipFilterInteractiveClassName =
  "transition duration-200 ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent-strong)_38%,var(--line))] hover:bg-[linear-gradient(180deg,color-mix(in_srgb,white_90%,var(--accent)_10%),color-mix(in_srgb,var(--tag-filter-fill-strong)_88%,white_12%))] hover:text-[var(--accent-strong)] hover:shadow-[inset_0_1px_0_color-mix(in_srgb,white_75%,transparent),0_14px_24px_color-mix(in_srgb,var(--accent-strong)_12%,transparent)] focus-visible:border-[color:color-mix(in_srgb,var(--accent-strong)_40%,var(--line))] focus-visible:outline-none focus-visible:shadow-[0_0_0_0.2rem_color-mix(in_srgb,var(--accent)_16%,transparent),0_14px_24px_color-mix(in_srgb,var(--accent-strong)_12%,transparent)]";

const tagChipMutedClassName =
  "border border-[color:color-mix(in_srgb,var(--ink)_10%,transparent)] bg-[linear-gradient(180deg,color-mix(in_srgb,white_92%,var(--paper-strong)_8%),color-mix(in_srgb,var(--tag-muted-fill)_92%,white_8%))] text-[color:color-mix(in_srgb,var(--ink-soft)_92%,var(--ink)_8%)] shadow-[inset_0_1px_0_color-mix(in_srgb,white_82%,transparent)] before:bg-[color:color-mix(in_srgb,var(--paper-strong)_72%,var(--ink-soft)_28%)]";

type TagChipProps = {
  label: string;
  href?: string;
  muted?: boolean;
  className?: string;
};

export function TagChip({ label, href, muted = false, className }: TagChipProps) {
  const tagChipClassName = clsx(
    tagChipBaseClassName,
    muted ? tagChipMutedClassName : tagChipFilterClassName,
    href && !muted && tagChipFilterInteractiveClassName,
    className
  );

  if (href) {
    return (
      <Link href={href} className={tagChipClassName}>
        <span className="ml-[0.18rem]">{label}</span>
      </Link>
    );
  }

  return (
    <span className={tagChipClassName}>
      <span className="ml-[0.18rem]">{label}</span>
    </span>
  );
}
