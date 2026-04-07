import type { Metadata } from "next";
import Image from "next/image";
import type { SVGProps } from "react";

import { MotionReveal } from "@/components/motion-reveal";
import { siteConfig } from "@/lib/site-config";

export const metadata: Metadata = {
  title: "About",
  description: "블로그 소개와 작성자 정보를 확인할 수 있습니다."
};

const authorLinks = [
  {
    href: `mailto:${siteConfig.author.email}`,
    label: "Email",
    icon: "email" as const
  },
  {
    href: "https://github.com/ddangchani",
    label: "GitHub",
    icon: "github" as const,
    external: true
  },
  {
    href: "https://www.linkedin.com/in/dangchan-kim-13932b213/",
    label: "LinkedIn",
    icon: "linkedin" as const,
    external: true
  }
];

type TimelineEntry = {
  period: string;
  title: string;
  subtitle?: string;
  details?: string[];
};

const educationEntries: TimelineEntry[] = [
  {
    period: "2022.03 ~",
    title: "Seoul National University, Department of Statistics",
    subtitle: "M.S. in Statistics",
    details: ["Spatial Statistics Lab."]
  },
  {
    period: "2021.02",
    title: "Korea National Police University",
    details: ["B.A. in Public Administration", "B.A. in Police Science"]
  }
];

const experienceEntries: TimelineEntry[] = [
  {
    period: "2021.03 ~",
    title: "Korea National Police Agency",
    subtitle: "Police Officer, Inspector",
    details: [ "AI Development Team - 2025.12 ~" , "Platoon Leader of Auxiliary Police, Seoul Metropolitan Police Agency",]
  },
  {
    period: "2023 ~ 2024",
    title: "Seoul National University",
    subtitle: "Teaching Assistant",
    details: [
      "Statistics Lab - Spring 2023",
      "Introduction to Data Science - Fall 2023, Fall 2024",
      "LG Electronics Data Science Course - 2024.01",
      "Samsung Electronics Data Science Course - 2024.02",
      "Statistical Computing - Spring 2024"
    ]
  }
];

type AboutIconName = "email" | "github" | "linkedin" | "location";

function AboutIcon({ name, ...props }: SVGProps<SVGSVGElement> & { name: AboutIconName }) {
  switch (name) {
    case "location":
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" {...props}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 21s6-5.4 6-11a6 6 0 1 0-12 0c0 5.6 6 11 6 11Z" />
          <circle cx="12" cy="10" r="2.4" />
        </svg>
      );
    case "email":
      return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" {...props}>
          <rect x="3.25" y="5.25" width="17.5" height="13.5" rx="2.5" />
          <path strokeLinecap="round" strokeLinejoin="round" d="m5.5 8.25 6.5 5 6.5-5" />
        </svg>
      );
    case "github":
      return (
        <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
          <path d="M12 .75a11.25 11.25 0 0 0-3.56 21.92c.56.1.76-.24.76-.54v-2.1c-3.09.67-3.74-1.31-3.74-1.31-.5-1.27-1.24-1.6-1.24-1.6-1-.68.08-.67.08-.67 1.12.08 1.7 1.14 1.7 1.14.98 1.7 2.6 1.2 3.23.92.1-.73.4-1.2.72-1.48-2.47-.28-5.06-1.23-5.06-5.47 0-1.2.42-2.18 1.12-2.95-.12-.28-.49-1.42.1-2.95 0 0 .93-.3 3.02 1.13a10.47 10.47 0 0 1 5.5 0c2.1-1.43 3.02-1.13 3.02-1.13.6 1.53.23 2.67.11 2.95.7.77 1.12 1.75 1.12 2.95 0 4.25-2.6 5.18-5.08 5.45.41.35.77 1.04.77 2.1v3.11c0 .3.2.65.77.54A11.25 11.25 0 0 0 12 .75Z" />
        </svg>
      );
    case "linkedin":
      return (
        <svg viewBox="0 0 24 24" fill="currentColor" {...props}>
          <path d="M4.98 3.5a1.74 1.74 0 1 1 0 3.49 1.74 1.74 0 0 1 0-3.48ZM3.5 8.5h2.96V20.5H3.5V8.5Zm4.82 0h2.84v1.64h.05c.4-.75 1.36-1.93 3.23-1.93 3.46 0 4.1 2.28 4.1 5.24v7.05H15.6v-6.25c0-1.49-.03-3.4-2.07-3.4-2.08 0-2.4 1.63-2.4 3.3v6.35H8.32V8.5Z" />
        </svg>
      );
  }
}

function TimelineSection({
  kicker,
  title,
  entries
}: {
  kicker: string;
  title: string;
  entries: TimelineEntry[];
}) {
  return (
    <section className="grid gap-[1.1rem]">
      <div className="grid gap-[0.35rem]">
        <p className="m-0 text-[0.7rem] uppercase tracking-[0.2em] text-[var(--ink-soft)]">{kicker}</p>
        <h2 className="m-0 font-[var(--font-display)] text-[clamp(1.45rem,2.3vw,1.9rem)] leading-[1.12] tracking-[-0.03em] text-[var(--ink)]">
          {title}
        </h2>
      </div>
      <div className="grid gap-4">
        {entries.map((entry) => (
          <article
            key={`${title}-${entry.period}-${entry.title}`}
            className="grid gap-4 rounded-[calc(var(--radius-lg)+0.12rem)] border border-[color:color-mix(in_srgb,var(--ink)_8%,transparent)] bg-[linear-gradient(180deg,color-mix(in_srgb,white_90%,var(--paper-strong)_10%),color-mix(in_srgb,white_97%,var(--surface)_3%))] p-[clamp(1rem,1.8vw,1.3rem)] shadow-[0_10px_24px_color-mix(in_srgb,var(--ink)_5%,transparent)] md:grid-cols-[minmax(0,8.2rem)_minmax(0,1fr)]"
          >
            <div className="flex items-start">
              <span className="inline-flex rounded-full border border-[color:color-mix(in_srgb,var(--ink)_10%,transparent)] bg-[color:color-mix(in_srgb,var(--paper-strong)_35%,white)] px-3 py-[0.36rem] text-[0.72rem] font-medium tracking-[0.04em] text-[var(--ink-soft)]">
                {entry.period}
              </span>
            </div>
            <div className="grid gap-[0.7rem]">
              <div className="grid gap-[0.25rem]">
                <h3 className="m-0 text-[1.02rem] font-semibold leading-[1.45] text-[var(--ink)] [word-break:keep-all]">
                  {entry.title}
                </h3>
                {entry.subtitle ? (
                  <p className="m-0 text-[0.92rem] leading-[1.65] text-[var(--ink-soft)] [word-break:keep-all]">
                    {entry.subtitle}
                  </p>
                ) : null}
              </div>
              {entry.details?.length ? (
                <ul className="m-0 grid gap-[0.38rem] pl-[1.1rem] text-[0.93rem] leading-[1.72] text-[var(--ink)]">
                  {entry.details.map((detail) => (
                    <li key={detail} className="marker:text-[var(--accent-strong)] [word-break:keep-all]">
                      {detail}
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function AboutPage() {
  return (
    <div className="page-stack [gap:clamp(2.75rem,7vw,4.75rem)]">
      <MotionReveal className="mx-auto w-full max-w-[980px]">
        <section className="content-section [gap:clamp(1.25rem,3vw,2rem)]">
          <p className="section-kicker">Author</p>
          <div className="grid items-start gap-6 rounded-[calc(var(--radius-lg)+0.35rem)] border border-[var(--line)] bg-[linear-gradient(135deg,color-mix(in_srgb,white_76%,var(--paper-strong)_24%),color-mix(in_srgb,white_92%,var(--surface)_8%)),radial-gradient(circle_at_100%_0%,color-mix(in_srgb,var(--accent)_10%,transparent),transparent_34%)] p-[clamp(1.35rem,3vw,2.2rem)] shadow-[var(--shadow)] md:grid-cols-[minmax(0,240px)_minmax(0,1fr)] md:gap-[clamp(1.5rem,4vw,3.25rem)]">
            <div className="grid max-w-[240px] gap-3 md:max-w-none">
              <div className="relative rounded-[calc(var(--radius-lg)+0.15rem)] border border-[color:color-mix(in_srgb,var(--ink)_10%,transparent)] bg-[linear-gradient(180deg,color-mix(in_srgb,white_60%,var(--paper-strong)_40%),color-mix(in_srgb,white_92%,var(--surface)_8%))] p-[clamp(0.7rem,1.5vw,0.9rem)] after:pointer-events-none after:absolute after:inset-x-5 after:bottom-[-1rem] after:h-[1.4rem] after:rounded-full after:bg-[color:color-mix(in_srgb,var(--accent-strong)_18%,transparent)] after:opacity-35 after:blur-[16px] after:content-['']">
                <Image
                  src="/assets/logos/bio-photo.png"
                  alt={siteConfig.author.fullName}
                  className="aspect-square w-full rounded-[1.55rem] border border-[color:color-mix(in_srgb,var(--ink)_10%,transparent)] object-cover shadow-[0_16px_30px_color-mix(in_srgb,var(--ink)_10%,transparent)]"
                  width={220}
                  height={220}
                />
              </div>
            </div>
            <div className="grid gap-[1.15rem]">
              <p className="m-0 text-[0.78rem] uppercase tracking-[0.18em] text-[var(--accent-strong)]">
                {siteConfig.author.role}
              </p>
              <h1 className="page-title text-[clamp(2.15rem,4vw,3.45rem)]">{siteConfig.author.fullName}</h1>
              <p className="m-0 max-w-[39rem] text-[clamp(1.08rem,2vw,1.32rem)] leading-[1.9] text-[var(--ink)] [word-break:keep-all]">
                AI와 데이터를 기반으로 더 안전한 사회와 더 나은 세상에 기여하는 일에 관심이 있습니다. <br />
                <strong>공공안전</strong> 관련 분야에서 AI·Data Science 활용과 관련된 연구와 실무 경험을 쌓아가고 있으며, 이를 바탕으로 다양한 지식과 인사이트를 나누고자 합니다.
              </p>
              <dl className="grid gap-x-[1.2rem] gap-y-[0.9rem] md:grid-cols-2">
                <div className="grid gap-[0.45rem] border-t border-[color:color-mix(in_srgb,var(--ink)_10%,transparent)] pt-[0.95rem]">
                  <dt className="m-0 text-[0.72rem] uppercase tracking-[0.18em] text-[var(--ink-soft)]">Base</dt>
                  <dd className="m-0 flex items-center gap-[0.7rem] text-[var(--ink)]">
                    <AboutIcon name="location" className="h-[1.1rem] w-[1.1rem] shrink-0 text-[var(--accent-strong)]" aria-hidden="true" />
                    <span>{siteConfig.author.location}</span>
                  </dd>
                </div>
                <div className="grid gap-[0.45rem] border-t border-[color:color-mix(in_srgb,var(--ink)_10%,transparent)] pt-[0.95rem]">
                  <dt className="m-0 text-[0.72rem] uppercase tracking-[0.18em] text-[var(--ink-soft)]">
                    Reach me
                  </dt>
                  <dd className="m-0 flex items-center gap-[0.7rem] text-[var(--ink)]">
                    <AboutIcon name="email" className="h-[1.1rem] w-[1.1rem] shrink-0 text-[var(--accent-strong)]" aria-hidden="true" />
                    <span>{siteConfig.author.email}</span>
                  </dd>
                </div>
              </dl>
              <div className="mt-[0.3rem] flex flex-wrap gap-3" aria-label="Author links">
                {authorLinks.map((link) => (
                  <a
                    href={link.href}
                    key={link.label}
                    className="inline-flex w-full items-center justify-center gap-[0.65rem] rounded-full border border-[var(--line)] bg-[color:color-mix(in_srgb,white_88%,var(--surface)_12%)] px-4 py-[0.78rem] transition duration-200 ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_30%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_10%,white)] sm:w-auto"
                    rel={link.external ? "noreferrer" : undefined}
                    target={link.external ? "_blank" : undefined}
                  >
                    <AboutIcon
                      name={link.icon}
                      className="h-[1.1rem] w-[1.1rem] shrink-0 text-[var(--accent-strong)]"
                      aria-hidden="true"
                    />
                    <span>{link.label}</span>
                  </a>
                ))}
              </div>
            </div>
          </div>
        </section>
      </MotionReveal>
      <MotionReveal delay={0.08} className="mx-auto w-full max-w-[980px]">
        <section className="content-section content-section--split">
          <TimelineSection kicker="Education" title="Education" entries={educationEntries} />
          <TimelineSection kicker="Experience" title="Experience" entries={experienceEntries} />
        </section>
      </MotionReveal>
    </div>
  );
}
