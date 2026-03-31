"use client";

import { useEffect, useId } from "react";

import { siteConfig } from "@/lib/site-config";

type PostCommentsProps = {
  pathname: string;
};

export function PostComments({ pathname }: PostCommentsProps) {
  const containerId = useId().replace(/:/g, "");

  useEffect(() => {
    const container = document.getElementById(containerId);

    if (!container) {
      return;
    }

    container.innerHTML = "";

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

    return () => {
      container.innerHTML = "";
    };
  }, [containerId, pathname]);

  return <div id={containerId} className="post-comments" />;
}
