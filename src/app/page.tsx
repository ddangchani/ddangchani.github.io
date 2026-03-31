import Link from "next/link";

import { EmptyArchive } from "@/components/empty-archive";
import { Hero } from "@/components/hero";
import { MotionReveal } from "@/components/motion-reveal";
import { PostCard } from "@/components/post-card";
import { SectionHeading } from "@/components/section-heading";
import { getAllTags, getFeaturedPosts, getPopularEntries, getRecentPosts, getTopicSpotlight } from "@/lib/site-data";

export default async function HomePage() {
  const [featuredPosts, recentPosts, popularEntries, tags, spotlight] = await Promise.all([
    getFeaturedPosts(),
    getRecentPosts(6),
    getPopularEntries(),
    getAllTags(),
    getTopicSpotlight()
  ]);

  return (
    <div className="page-stack">
      <Hero postCount={recentPosts.length} tagCount={tags.length} />

      <MotionReveal>
        <section className="content-section content-section--wide">
          <SectionHeading
            eyebrow="Featured"
            title="Research notes worth entering from the front door"
            description="Recent essays and cornerstone posts surface here first, with a structure tuned for study rather than a feed."
            action={<Link href="/posts/">Browse all posts</Link>}
          />
          {featuredPosts.length > 0 ? (
            <div className="post-grid post-grid--feature">
              {featuredPosts.map((post) => (
                <PostCard key={post.route} post={post} />
              ))}
            </div>
          ) : (
            <EmptyArchive
              title="Featured posts will appear here once generated content is available."
              description="The frontend is ready. It is waiting for the content pipeline to emit the shared post manifest."
            />
          )}
        </section>
      </MotionReveal>

      <MotionReveal delay={0.08}>
        <section className="content-section content-section--split">
          <div>
            <SectionHeading
              eyebrow="Topics"
              title="A live map of the archive"
              description="The most active subjects rise to the top, useful for both deep dives and rediscovering old series."
            />
            <div className="topic-cloud">
              {spotlight.map((topic) => (
                <Link key={topic.title} href={`/tags/${encodeURIComponent(topic.title)}/`} className="topic-cloud__item">
                  <strong>{topic.title}</strong>
                  <span>{topic.count} notes</span>
                </Link>
              ))}
            </div>
          </div>
          <div>
            <SectionHeading
              eyebrow="Popular"
              title="Frequently revisited notes"
              description="Popularity data stays in the loop, but the UI treats it as a reading signal rather than clickbait."
            />
            <ul className="popular-list">
              {popularEntries.slice(0, 5).map((entry) => (
                <li key={entry.route} className="popular-list__item">
                  <a href={entry.route} className="popular-list__link">
                    <span className="popular-list__count">{entry.viewCount}</span>
                    <span className="popular-list__content">
                      <strong>{entry.title ?? entry.route}</strong>
                      <span>{entry.excerpt ?? "Popularity data is present; waiting on richer content metadata."}</span>
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </section>
      </MotionReveal>

      <MotionReveal delay={0.14}>
        <section className="content-section">
          <SectionHeading
            eyebrow="Recent"
            title="Latest additions to the notebook"
            description="Chronology stays accessible, but the layout shifts away from the old Jekyll list into a more scan-friendly reading deck."
          />
          {recentPosts.length > 0 ? (
            <div className="post-grid">
              {recentPosts.map((post) => (
                <PostCard key={post.route} post={post} compact />
              ))}
            </div>
          ) : (
            <EmptyArchive
              title="Recent posts will appear after the generated post index lands."
              description="This section is already wired to the shared artifact path and will populate without frontend changes."
            />
          )}
        </section>
      </MotionReveal>
    </div>
  );
}
