import { z } from "zod";

export const routeSegmentsSchema = z.array(z.string().trim().min(1)).min(1);

export const postMetaSchema = z.object({
  title: z.string().trim().min(1),
  description: z.string().trim().min(1),
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  tags: z.array(z.string().trim().min(1)),
  categories: z.array(z.string().trim().min(1)),
  routeSegments: routeSegmentsSchema,
  teaser: z.string().trim().min(1).nullable(),
  coverImage: z.string().trim().min(1).nullable(),
  excerpt: z.string().trim().min(1),
  readingTimeMinutes: z.number().int().positive(),
  useMath: z.boolean(),
  featured: z.boolean(),
  draft: z.boolean(),
  canonicalUrl: z.url()
});

export const searchEntrySchema = z.object({
  title: z.string().trim().min(1),
  route: z.string().trim().min(1),
  description: z.string().trim().min(1),
  tags: z.array(z.string().trim().min(1)),
  categories: z.array(z.string().trim().min(1)),
  bodyPlain: z.string().trim().min(1),
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  language: z.enum(["ko", "en", "mixed"])
});

export const popularPostEntrySchema = z.object({
  route: z.string().trim().min(1),
  viewCount: z.number().int().nonnegative(),
  title: z.string().trim().min(1),
  teaser: z.string().trim().min(1).nullable(),
  excerpt: z.string().trim().min(1)
});
