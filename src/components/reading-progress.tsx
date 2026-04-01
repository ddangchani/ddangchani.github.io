"use client";

import { useEffect, useState } from "react";

export function ReadingProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    function onScroll() {
      const totalHeight = document.documentElement.scrollHeight - window.innerHeight;

      if (totalHeight <= 0) {
        setProgress(0);
        return;
      }

      setProgress((window.scrollY / totalHeight) * 100);
    }

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    return () => {
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  return (
    <div
      className="sticky top-[4.65rem] z-20 h-[0.35rem] w-full overflow-hidden rounded-full bg-[color:color-mix(in_srgb,var(--ink)_8%,transparent)]"
      aria-hidden="true"
    >
      <span
        className="block h-full w-full origin-left bg-[linear-gradient(90deg,var(--accent),var(--accent-strong))]"
        style={{ transform: `scaleX(${progress / 100})` }}
      />
    </div>
  );
}
