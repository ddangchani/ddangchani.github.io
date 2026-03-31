import Link from "next/link";

import { siteConfig } from "@/lib/site-config";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer__inner">
        <div>
          <p className="site-footer__label">Archive</p>
          <p className="site-footer__text">{siteConfig.description}</p>
        </div>
        <div className="site-footer__links" aria-label="External links">
          {siteConfig.links.map((link) => (
            <Link key={link.href} href={link.href} className="site-footer__link">
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </footer>
  );
}
