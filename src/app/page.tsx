import Link from "next/link";

import { EmptyArchive } from "@/components/empty-archive";
import { Hero } from "@/components/hero";
import { MotionReveal } from "@/components/motion-reveal";
import { PopularPostsCarousel } from "@/components/popular-posts-carousel";
import { SectionHeading } from "@/components/section-heading";
import { getAllPosts, getAllTags, getPopularEntries, getTopicSpotlight } from "@/lib/site-data";

const homeIndexLinkClassName =
  "relative inline-flex min-h-[2.9rem] items-center gap-[0.7rem] rounded-full border border-[color:color-mix(in_srgb,var(--accent-strong)_18%,var(--line))] bg-[color:color-mix(in_srgb,white_86%,var(--paper-strong)_14%)] px-4 py-[0.6rem] pl-[1.15rem] text-[var(--ink)] shadow-[0_14px_34px_color-mix(in_srgb,var(--accent-strong)_9%,transparent)] transition duration-200 ease-out before:h-[0.55rem] before:w-[0.55rem] before:rounded-full before:bg-[linear-gradient(135deg,color-mix(in_srgb,var(--accent)_72%,white),var(--accent-strong))] before:shadow-[0_0_0_0.28rem_color-mix(in_srgb,var(--accent)_10%,transparent)] after:text-[0.82rem] after:tracking-[0.08em] after:text-[var(--accent-strong)] after:content-['->'] hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent-strong)_36%,var(--line))] hover:bg-[color:color-mix(in_srgb,white_78%,var(--paper-strong)_22%)] hover:shadow-[0_18px_42px_color-mix(in_srgb,var(--accent-strong)_13%,transparent)]";

export default async function HomePage() {
  const [allPosts, popularEntries, tags, spotlight] = await Promise.all([
    getAllPosts(),
    getPopularEntries(),
    getAllTags(),
    getTopicSpotlight()
  ]);
  const recentPosts = allPosts.slice(0, 4);
  const featuredPopularEntries = popularEntries.slice(0, 5);

  return (
    <div className="page-stack page-stack--home">
      <Hero postCount={allPosts.length} tagCount={tags.length} />

      <MotionReveal>
        <section className="content-section content-section--wide">
          <SectionHeading
            eyebrow="Main"
            title="Popular Posts"
            action={
              <Link href="/posts/" className={homeIndexLinkClassName}>
                Browse all articles
              </Link>
            }
          />
          {featuredPopularEntries.length > 0 ? (
            <PopularPostsCarousel entries={featuredPopularEntries} />
          ) : (
            <EmptyArchive
              title="Popular posts will appear here once analytics data is available."
              description="홈 상단은 예전 메인처럼 인기 글을 가장 먼저 보여주도록 연결되어 있습니다."
            />
          )}
        </section>
      </MotionReveal>

      <MotionReveal delay={0.08}>
        <section className="content-section content-section--split min-[961px]:items-stretch">
          <div className="grid h-full min-h-0 grid-rows-[auto_1fr] gap-[clamp(0.9rem,2vw,1.25rem)]">
            <SectionHeading
              eyebrow="Archive"
              title="Latest Articles"
              className="gap-3"
              action={
                <Link href="/posts/" className={homeIndexLinkClassName}>
                  See full archive
                </Link>
              }
            />
            {recentPosts.length > 0 ? (
              <div className="grid h-full auto-rows-fr gap-3 sm:grid-cols-2 min-[961px]:grid-cols-3 min-[1180px]:grid-cols-4">
                {recentPosts.map((post) => (
                  <article
                    key={post.route}
                    className="grid h-full min-h-[13.5rem] grid-rows-[auto_auto_1fr_auto] gap-[0.85rem] overflow-hidden rounded-[1.5rem] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_90%,white)] p-[1.1rem] shadow-[var(--shadow)] transition duration-200 ease-out hover:-translate-y-[2px] hover:border-[color:color-mix(in_srgb,var(--accent-strong)_28%,var(--line))] hover:bg-[color:color-mix(in_srgb,white_84%,var(--paper-strong)_16%)] hover:shadow-[0_20px_46px_color-mix(in_srgb,var(--accent-strong)_11%,transparent)] max-[720px]:min-h-[12rem] max-[720px]:p-4"
                  >
                    <div className="flex flex-wrap gap-x-[0.75rem] gap-y-[0.35rem] text-[0.76rem] uppercase tracking-[0.08em] text-[var(--ink-soft)]">
                      <span>{new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium" }).format(new Date(post.date))}</span>
                      <span>{post.categories[0] ?? "Archive"}</span>
                    </div>
                    <h3 className="m-0 text-[1.08rem] leading-[1.22] [overflow-wrap:anywhere]">
                      <Link href={post.route} className="transition-colors duration-200 hover:text-[var(--accent-strong)]">
                        {post.title}
                      </Link>
                    </h3>
                    <p className="m-0 text-[0.92rem] leading-[1.65] text-[var(--ink-soft)]">
                      {post.description || post.excerpt}
                    </p>
                    <div className="flex items-end justify-between gap-3 pt-1">
                      <Link
                        href={post.route}
                        className="inline-flex items-center gap-2 text-[0.88rem] font-medium text-[var(--accent-strong)] transition duration-200 ease-out hover:translate-x-[2px]"
                      >
                        Read note
                        <span aria-hidden="true">{"->"}</span>
                      </Link>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <EmptyArchive
                title="Recent posts will appear here after content generation."
                description="정적 콘텐츠 인덱스가 준비되면 최신 글 목록이 자동으로 채워집니다."
              />
            )}
          </div>
          <aside className="grid h-full content-start gap-[0.95rem] overflow-hidden rounded-[var(--radius-lg)] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_92%,white)] p-[clamp(1.2rem,3vw,1.7rem)] shadow-[var(--shadow)]">
            <SectionHeading
              eyebrow="Explore"
              title="Topics"
              description="자주 다루는 주제를 빠르게 훑고, 관심 있는 주제에서 바로 글을 이어서 읽을 수 있습니다."
            />
            <div className="grid gap-[0.9rem] min-[481px]:grid-cols-2 min-[961px]:grid-cols-1">
              {spotlight.map((topic) => (
                <Link
                  key={topic.title}
                  href={`/tags/${topic.title}/`}
                  className="group relative grid gap-[0.35rem] overflow-hidden rounded-[1.4rem] border border-[var(--line)] bg-[color:color-mix(in_srgb,var(--surface)_88%,white)] px-[1.15rem] py-4 transition duration-200 ease-out after:absolute after:right-[1.15rem] after:bottom-[0.9rem] after:left-[1.15rem] after:h-[2px] after:origin-left after:scale-x-[0.18] after:rounded-full after:bg-[linear-gradient(90deg,color-mix(in_srgb,var(--accent)_70%,white),transparent)] after:opacity-65 after:transition-all after:duration-200 after:ease-out hover:-translate-y-[2px] hover:border-[color:color-mix(in_srgb,var(--accent-strong)_30%,var(--line))] hover:bg-[color:color-mix(in_srgb,white_82%,var(--paper-strong)_18%)] hover:shadow-[0_18px_42px_color-mix(in_srgb,var(--accent-strong)_10%,transparent)] hover:after:scale-x-100 hover:after:opacity-100 max-[720px]:px-4 max-[720px]:py-[0.95rem]"
                >
                  <strong>{topic.title}</strong>
                  <span className="text-[0.86rem] text-[var(--ink-soft)]">{topic.count} notes</span>
                </Link>
              ))}
            </div>
            <Link href="/tags/" className={`${homeIndexLinkClassName} mt-1 justify-self-start text-[var(--accent-strong)]`}>
              Browse all topics
            </Link>
          </aside>
        </section>
      </MotionReveal>
    </div>
  );
}
