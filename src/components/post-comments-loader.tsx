"use client";

import { useEffect } from "react";

import { POST_COMMENTS_CONTAINER_ID } from "@/components/post-comments-shared";
import { siteConfig } from "@/lib/site-config";

type PostCommentsLoaderProps = {
  pathname: string;
};

export function PostCommentsLoader({ pathname }: PostCommentsLoaderProps) {
  useEffect(() => {
    const container = document.getElementById(POST_COMMENTS_CONTAINER_ID);

    if (!container) {
      return;
    }

    let cancelled = false;

    const observer = new MutationObserver(() => {
      const iframe = container.querySelector("iframe");

      if (iframe) {
        iframe.setAttribute("loading", "eager");
        observer.disconnect();
      }
    });

    const frame = window.requestAnimationFrame(() => {
      if (cancelled || !container.isConnected) {
        return;
      }

      const existingIframe = container.querySelector("iframe");
      const existingPathname = container.getAttribute("data-comments-pathname");

      if (existingIframe && existingPathname === pathname) {
        existingIframe.setAttribute("loading", "eager");
        return;
      }

      observer.observe(container, { childList: true, subtree: true });
      container.replaceChildren();
      container.setAttribute("data-comments-pathname", pathname);

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
      observer.disconnect();
    };
  }, [pathname]);

  return <span aria-hidden="true" className="block h-0 w-0 overflow-hidden" />;
}
