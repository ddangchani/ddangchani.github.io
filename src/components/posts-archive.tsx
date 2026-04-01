"use client";

import { clsx } from "clsx";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { startTransition } from "react";

import { EmptyArchive } from "@/components/empty-archive";
import { PostCard } from "@/components/post-card";
import { TagFilterButton } from "@/components/tag-filter-button";
import { isFilterTag } from "@/lib/filter-tags";
import { encodeTagQueryValue, tagsMatch } from "@/lib/tag-query";

type ArchivePost = {
  route: string;
  title: string;
  description: string;
  excerpt: string;
  date: string;
  categories: string[];
  tags: string[];
  teaser: string | null;
};

type TagSummary = {
  name: string;
  count: number;
};

type PostsArchiveProps = {
  posts: ArchivePost[];
  tags: TagSummary[];
};

export function PostsArchive({ posts, tags }: PostsArchiveProps) {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedParams = searchParams
    .getAll("tag")
    .filter((selected) => tags.some((tag) => tagsMatch(tag.name, selected)));
  const activeTags = tags.filter((tag) => selectedParams.some((selected) => tagsMatch(tag.name, selected)));
  const filteredPosts =
    selectedParams.length === 0
      ? posts
      : posts.filter((post) =>
          post.tags.some(
            (postTag) =>
              isFilterTag(postTag) && selectedParams.some((selected) => tagsMatch(postTag, selected))
          )
        );

  function replaceTags(nextValues: string[]) {
    const nextParams = new URLSearchParams(searchParams.toString());
    nextParams.delete("tag");

    for (const value of nextValues) {
      nextParams.append("tag", value);
    }

    const nextQuery = nextParams.toString();
    const href = nextQuery ? `${pathname}?${nextQuery}` : pathname;

    startTransition(() => {
      router.replace(href, { scroll: false });
    });
  }

  function handleToggleTag(tagName: string) {
    const canonicalValue = encodeTagQueryValue(tagName);
    const remaining = selectedParams.filter((selected) => !tagsMatch(tagName, selected));

    if (remaining.length === selectedParams.length) {
      replaceTags([...remaining, canonicalValue]);
      return;
    }

    replaceTags(remaining);
  }

  function clearAll() {
    replaceTags([]);
  }

  return (
    <div className="grid gap-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="grid gap-[0.35rem]">
          <p className="m-0 [font-family:var(--font-display),serif] text-[clamp(1.45rem,3vw,2rem)] leading-[1.1]">
            {filteredPosts.length} posts
          </p>
          <p className="m-0 leading-[1.7] text-[var(--ink-soft)] [word-break:keep-all]">
            {selectedParams.length > 0
              ? `${
                  activeTags.length > 0
                    ? `${activeTags.map((tag) => tag.name).join(", ")} 태그 기준으로 글을 보고 있습니다.`
                    : "선택한 태그 기준으로 글을 필터링하고 있습니다."
                }`
              : "태그를 선택해서 원하는 주제의 글만 빠르게 골라볼 수 있습니다."}
          </p>
        </div>
        {selectedParams.length > 0 ? (
          <button
            type="button"
            className="inline-flex w-full items-center justify-center rounded-full border border-[var(--line)] bg-[color:color-mix(in_srgb,white_90%,var(--surface)_10%)] px-[0.95rem] py-[0.7rem] transition duration-200 ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_32%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_10%,white)] md:w-auto"
            onClick={clearAll}
          >
            Clear all
          </button>
        ) : null}
      </div>

      <div className="relative flex flex-wrap gap-3 rounded-[calc(var(--radius-lg)-0.15rem)] border border-[color:color-mix(in_srgb,var(--accent-strong)_12%,var(--line))] bg-[radial-gradient(circle_at_0%_0%,color-mix(in_srgb,var(--accent)_10%,transparent),transparent_26%),radial-gradient(circle_at_100%_100%,color-mix(in_srgb,var(--accent-strong)_8%,transparent),transparent_30%),linear-gradient(180deg,color-mix(in_srgb,white_90%,var(--surface)_10%),color-mix(in_srgb,white_98%,var(--surface)_2%))] p-4 shadow-[inset_0_1px_0_color-mix(in_srgb,white_78%,transparent),0_22px_42px_color-mix(in_srgb,var(--accent-strong)_6%,transparent)] max-[720px]:p-[0.85rem]" aria-label="Tag filters">
        {tags.map((tag) => {
          const isActive = activeTags.some((activeTag) => activeTag.name === tag.name);

          return (
            <TagFilterButton
              key={tag.name}
              name={tag.name}
              count={tag.count}
              isActive={isActive}
              onClick={() => handleToggleTag(tag.name)}
            />
          );
        })}
      </div>

      {filteredPosts.length > 0 ? (
        <div className="grid grid-cols-1 gap-[1.3rem] md:grid-cols-2">
          {filteredPosts.map((post) => (
            <PostCard key={post.route} post={post} compact />
          ))}
        </div>
      ) : (
        <div className="grid justify-items-start gap-4">
          <EmptyArchive
            title="선택한 태그에 맞는 글이 아직 없습니다."
            description="다른 태그를 추가로 고르거나 필터를 초기화해서 전체 아카이브를 다시 살펴보세요."
          />
          <button
            type="button"
            className="inline-flex w-full items-center justify-center rounded-full border border-[var(--line)] bg-[color:color-mix(in_srgb,white_90%,var(--surface)_10%)] px-[0.95rem] py-[0.7rem] transition duration-200 ease-out hover:-translate-y-px hover:border-[color:color-mix(in_srgb,var(--accent)_32%,var(--line))] hover:bg-[color:color-mix(in_srgb,var(--accent)_10%,white)] md:w-auto"
            onClick={clearAll}
          >
            Reset filters
          </button>
        </div>
      )}
    </div>
  );
}
