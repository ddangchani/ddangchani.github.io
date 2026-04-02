import { clsx } from "clsx";

const tagFilterButtonBaseClassName =
  "relative inline-flex w-auto items-center gap-[0.48rem] whitespace-nowrap rounded-full px-[0.72rem] py-[0.35rem] pl-[1.2rem] font-medium shadow-[inset_0_1px_0_color-mix(in_srgb,white_80%,transparent),0_10px_20px_color-mix(in_srgb,var(--ink)_4%,transparent)] transition duration-200 ease-out before:absolute before:left-[0.72rem] before:top-1/2 before:h-[0.38rem] before:w-[0.38rem] before:-translate-y-1/2 before:rounded-full";

const tagFilterButtonInactiveClassName =
  "border border-[color:color-mix(in_srgb,var(--accent-strong)_10%,var(--line))] bg-[linear-gradient(180deg,color-mix(in_srgb,white_96%,var(--paper-strong)_4%),color-mix(in_srgb,white_90%,var(--paper-strong)_10%))] text-[var(--ink)] before:bg-[color:color-mix(in_srgb,var(--accent)_56%,white)] before:shadow-[0_0_0_0.24rem_color-mix(in_srgb,var(--accent)_10%,transparent)] hover:-translate-y-[2px] hover:border-[color:color-mix(in_srgb,var(--accent)_28%,var(--line))] hover:shadow-[inset_0_1px_0_color-mix(in_srgb,white_82%,transparent),0_14px_26px_color-mix(in_srgb,var(--accent-strong)_8%,transparent)] focus-visible:border-[color:color-mix(in_srgb,var(--accent-strong)_38%,var(--line))] focus-visible:outline-none focus-visible:shadow-[0_0_0_0.2rem_color-mix(in_srgb,var(--accent)_16%,transparent),0_14px_26px_color-mix(in_srgb,var(--accent-strong)_10%,transparent)]";

const tagFilterButtonActiveClassName =
  "border-[var(--tag-filter-active-line)] bg-[radial-gradient(circle_at_0%_0%,color-mix(in_srgb,white_24%,var(--accent)_76%),transparent_42%),linear-gradient(180deg,var(--tag-filter-active-fill),var(--tag-filter-active-fill-strong))] text-[color:color-mix(in_srgb,var(--accent-strong)_82%,var(--ink)_18%)] shadow-[inset_0_1px_0_color-mix(in_srgb,white_62%,transparent),0_16px_30px_color-mix(in_srgb,var(--accent-strong)_18%,transparent)] before:bg-[var(--accent-strong)] before:shadow-[0_0_0_0.3rem_color-mix(in_srgb,var(--accent)_20%,transparent)]";

const tagFilterCountBaseClassName =
  "min-w-[1.35rem] rounded-full border px-[0.36rem] py-[0.1rem] text-[0.72rem] shadow-[inset_0_1px_0_color-mix(in_srgb,white_72%,transparent)]";

const tagFilterCountInactiveClassName =
  "border-[color:color-mix(in_srgb,var(--accent-strong)_10%,transparent)] bg-[color:color-mix(in_srgb,white_86%,var(--paper-strong)_14%)] text-[var(--ink-soft)]";

const tagFilterCountActiveClassName =
  "border-[color:color-mix(in_srgb,var(--accent-strong)_28%,transparent)] bg-[var(--tag-filter-active-count-fill)] text-[color:color-mix(in_srgb,var(--accent-strong)_84%,var(--ink)_16%)]";

type TagFilterButtonProps = {
  name: string;
  count: number;
  isActive: boolean;
  onClick: () => void;
};

export function TagFilterButton({ name, count, isActive, onClick }: TagFilterButtonProps) {
  return (
    <button
      type="button"
      className={clsx(
        tagFilterButtonBaseClassName,
        tagFilterButtonInactiveClassName,
        isActive && tagFilterButtonActiveClassName
      )}
      data-active={isActive}
      aria-pressed={isActive}
      onClick={onClick}
    >
      <span className="ml-[0.42rem] text-[0.84rem]">{name}</span>
      <span
        className={clsx(
          tagFilterCountBaseClassName,
          tagFilterCountInactiveClassName,
          isActive && tagFilterCountActiveClassName
        )}
      >
        {count}
      </span>
    </button>
  );
}
