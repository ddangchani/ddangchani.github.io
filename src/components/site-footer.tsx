import Link from "next/link";

import { siteConfig } from "@/lib/site-config";

export function SiteFooter() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="mt-14 border-t border-[var(--line)] px-0 py-5">
      <div className="mx-auto flex w-[var(--content-width)] flex-col gap-2 text-[0.84rem] leading-6 text-[var(--ink-soft)] md:flex-row md:items-center md:justify-between">
        <p className="m-0">
          <span className="font-medium text-[var(--ink)]">{siteConfig.author.fullName}</span>
          <span className="px-2 text-[var(--line-strong)]">/</span>
          <span>{currentYear}</span>
          <span className="px-2 text-[var(--line-strong)]">/</span>
          <span>{siteConfig.author.role}</span>
        </p>
        <nav className="flex flex-wrap items-center gap-x-3 gap-y-1" aria-label="External links">
          {siteConfig.links.map((link, index) => (
            <span key={link.href} className="flex items-center gap-3">
              {index > 0 ? <span className="text-[var(--line-strong)]">/</span> : null}
              <Link
                href={link.href}
                className="transition-colors duration-200 hover:text-[var(--ink)]"
              >
                {link.label}
              </Link>
            </span>
          ))}
        </nav>
      </div>
    </footer>
  );
}
