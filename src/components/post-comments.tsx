"use client";

import { useEffect, useRef } from "react";

import { siteConfig } from "@/lib/site-config";

type PostCommentsProps = {
  pathname: string;
};

export function PostComments({ pathname }: PostCommentsProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;

    if (!container) {
      return;
    }

    let cancelled = false;

    const frame = window.requestAnimationFrame(() => {
      if (cancelled || !container.isConnected) {
        return;
      }

      container.replaceChildren();

      const script = document.createElement("script");
      script.src = "https://utteranc.es/client.js";
      script.async = true;
      script.crossOrigin = "anonymous";
      script.setAttribute("repo", siteConfig.utterances.repo);
      script.setAttribute("issue-term", siteConfig.utterances.issueTerm);
      script.setAttribute("label", siteConfig.utterances.label);
      script.setAttribute("theme", "github-light");
      script.setAttribute("pathname", pathname);
      container.appendChild(script);
    });

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
    };
  }, [pathname]);

  return <div ref={containerRef} className="min-h-12" />;
}
