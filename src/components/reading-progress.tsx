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
    <div className="reading-progress" aria-hidden="true">
      <span className="reading-progress__bar" style={{ transform: `scaleX(${progress / 100})` }} />
    </div>
  );
}
