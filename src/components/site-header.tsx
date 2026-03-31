import Link from "next/link";

import { siteConfig } from "@/lib/site-config";

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-header__bar">
        <Link href="/" className="site-header__brand" aria-label={siteConfig.title}>
          <span className="site-header__eyebrow">Technical Notebook</span>
          <span className="site-header__title">{siteConfig.title}</span>
        </Link>
        <nav className="site-header__nav" aria-label="Primary">
          {siteConfig.nav.map((item) => (
            <Link key={item.href} href={item.href} className="site-header__link">
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
