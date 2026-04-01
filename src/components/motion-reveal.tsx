"use client";

import { clsx } from "clsx";
import { motion, useReducedMotion } from "framer-motion";
import type { PropsWithChildren } from "react";

type MotionRevealProps = PropsWithChildren<{
  delay?: number;
  className?: string;
}>;

export function MotionReveal({ children, delay = 0, className }: MotionRevealProps) {
  const reducedMotion = useReducedMotion();
  const mergedClassName = clsx("motion-reveal", className);

  if (reducedMotion) {
    return <div className={mergedClassName}>{children}</div>;
  }

  return (
    <motion.div
      className={mergedClassName}
      initial={false}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.25 }}
      transition={{ duration: 0.72, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}
