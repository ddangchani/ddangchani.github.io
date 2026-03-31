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
    <div className={clsx("section-heading", className)}>
      <div>
        <p className="section-heading__eyebrow">{eyebrow}</p>
        <h2 className="section-heading__title">{title}</h2>
        {description ? <p className="section-heading__description">{description}</p> : null}
      </div>
      {action ? <div className="section-heading__action">{action}</div> : null}
    </div>
  );
}
