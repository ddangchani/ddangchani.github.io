import type { Metadata } from "next";
import { IBM_Plex_Mono, Noto_Sans_KR, Noto_Serif_KR } from "next/font/google";
import type { PropsWithChildren } from "react";

import { Analytics } from "@/components/analytics";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { siteConfig } from "@/lib/site-config";

import "katex/dist/katex.min.css";
import "./globals.css";

const bodyFont = Noto_Sans_KR({
  variable: "--font-body",
  weight: ["400", "500", "700"],
  subsets: ["latin"]
});

const displayFont = Noto_Serif_KR({
  variable: "--font-display",
  weight: ["400", "500", "700"],
  subsets: ["latin"]
});

const monoFont = IBM_Plex_Mono({
  variable: "--font-mono",
  weight: ["400", "500", "600"],
  subsets: ["latin"]
});

export const metadata: Metadata = {
  metadataBase: new URL(siteConfig.siteUrl),
  title: {
    default: siteConfig.title,
    template: `%s | ${siteConfig.title}`
  },
  description: siteConfig.description,
  openGraph: {
    type: "website",
    url: siteConfig.siteUrl,
    title: siteConfig.title,
    description: siteConfig.description
  },
  twitter: {
    card: "summary_large_image",
    title: siteConfig.title,
    description: siteConfig.description
  },
  icons: {
    icon: "/assets/logos/logo.ico",
    shortcut: "/assets/logos/logo.ico"
  }
};

export default function RootLayout({ children }: PropsWithChildren) {
  return (
    <html lang="ko">
      <body className={`${bodyFont.variable} ${displayFont.variable} ${monoFont.variable}`}>
        <Analytics trackingId={process.env.NEXT_PUBLIC_GA_ID ?? "G-QXX68H21RZ"} />
        <div className="site-frame">
          <SiteHeader />
          <main className="site-main">{children}</main>
          <SiteFooter />
        </div>
      </body>
    </html>
  );
}
