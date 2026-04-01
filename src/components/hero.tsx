import { MotionReveal } from "@/components/motion-reveal";
import { siteConfig } from "@/lib/site-config";

type HeroProps = {
  postCount: number;
  tagCount: number;
};

export function Hero({ postCount, tagCount }: HeroProps) {
  return (
    <section className="relative left-1/2 ml-[-50vw] w-screen overflow-hidden bg-[radial-gradient(circle_at_50%_0%,color-mix(in_srgb,white_10%,transparent),transparent_38%),linear-gradient(180deg,oklch(0.34_0.09_245),oklch(0.28_0.08_248))] text-[color:color-mix(in_srgb,white_92%,var(--paper)_8%)]">
      <MotionReveal className="mx-auto w-[var(--content-width)] pt-[clamp(4rem,8vw,6.5rem)] max-[720px]:pt-[3.4rem]">
        <div className="grid justify-items-center gap-4 px-[var(--page-gutter)] text-center max-[720px]:px-4">
          <p className="m-0 text-[0.74rem] uppercase tracking-[0.24em] text-[color:color-mix(in_srgb,white_70%,transparent)]">
            @ddangchan_i
          </p>
          <h1 className="mb-[5px] max-w-[12ch] [font-family:var(--font-display),serif] text-[clamp(3rem,7vw,5.1rem)] leading-[1.05] tracking-[-0.04em] text-white [overflow-wrap:anywhere] max-[720px]:text-[clamp(2.35rem,12vw,3.55rem)] max-[720px]:leading-none">
            {siteConfig.title}
          </h1>
          <p className="m-0 max-w-[38rem] text-[clamp(1.05rem,2vw,1.3rem)] leading-[1.9] text-[color:color-mix(in_srgb,white_82%,var(--paper)_18%)] [word-break:keep-all] max-[720px]:text-[0.97rem] max-[720px]:leading-[1.7]">
            {siteConfig.description}
          </p>
          <div className="mt-1 flex flex-wrap justify-center gap-3 pb-[clamp(2.5rem,6vw,4rem)] max-[480px]:gap-[0.55rem]">
            <span className="rounded-full border border-[color:color-mix(in_srgb,white_18%,transparent)] bg-[color:color-mix(in_srgb,white_9%,transparent)] px-[0.85rem] py-2 text-[0.84rem] text-[color:color-mix(in_srgb,white_82%,var(--paper)_18%)] max-[480px]:w-full">
              {postCount} notes archived
            </span>
            <span className="rounded-full border border-[color:color-mix(in_srgb,white_18%,transparent)] bg-[color:color-mix(in_srgb,white_9%,transparent)] px-[0.85rem] py-2 text-[0.84rem] text-[color:color-mix(in_srgb,white_82%,var(--paper)_18%)] max-[480px]:w-full">
              {tagCount} topics indexed
            </span>
          </div>
        </div>
      </MotionReveal>
      <div className="relative h-[clamp(4.75rem,10vw,7.75rem)] leading-none" aria-hidden="true">
        <div className="absolute inset-x-0 top-0 h-px bg-[color:color-mix(in_srgb,white_20%,transparent)]" />
        <svg
          className="absolute inset-0 block h-full w-full"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 1440 140"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="hero-paper-shadow" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="color-mix(in srgb, white 32%, transparent)" />
              <stop offset="100%" stopColor="transparent" />
            </linearGradient>
          </defs>
          <path
            d="M0 54C88 48 154 69 246 66C360 62 440 44 546 47C658 50 736 79 850 78C978 77 1052 46 1176 51C1289 56 1360 79 1440 75V140H0Z"
            fill="url(#hero-paper-shadow)"
          />
          <path
            d="M0 70C104 65 171 84 288 81C398 77 472 57 582 60C693 63 772 92 882 91C1006 89 1081 60 1208 64C1312 68 1379 92 1440 88V140H0Z"
            style={{ fill: "color-mix(in srgb, var(--paper) 95%, white)" }}
          />
          <path
            d="M0 70C104 65 171 84 288 81C398 77 472 57 582 60C693 63 772 92 882 91C1006 89 1081 60 1208 64C1312 68 1379 92 1440 88"
            fill="none"
            stroke="color-mix(in srgb, white 34%, transparent)"
            strokeWidth="1.5"
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      </div>
    </section>
  );
}
