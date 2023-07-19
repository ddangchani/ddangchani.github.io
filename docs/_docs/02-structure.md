---
title: "Structure"
permalink: /docs/structure/
excerpt: "How the theme is organized and what all of the files are for."
last_modified_at: 2018-03-20T15:19:22-04:00
---

Nothing clever here :wink:. Layouts, data files, and includes are all placed in their default locations. Stylesheets and scripts in `assets`, and a few development related files in the project's root directory.

**Please note:** If you installed Minimal Mistakes via the Ruby Gem method, theme files like `_layouts`, `_includes`, `_sass`, and `/assets/` will be missing. This is normal as they are bundled with the [`minimal-mistakes-jekyll`](https://rubygems.org/gems/minimal-mistakes-jekyll) Ruby gem. If you would like to make changes, create the files and Jekyll will prefer your local copy.
{: .notice--info}

```bash
minimal-mistakes
├── _data                      # data files for customizing the theme
\vert   ├── navigation.yml          # main navigation links
\vert   └── ui-text.yml             # text used throughout the theme's UI
├── _includes
\vert   ├── analytics-providers     # snippets for analytics (Google and custom)
\vert   ├── comments-providers      # snippets for comments
\vert   ├── footer
\vert   \vert   └── custom.html          # custom snippets to add to site footer
\vert   ├── head
\vert   \vert   └── custom.html          # custom snippets to add to site head
\vert   ├── feature_row             # feature row helper
\vert   ├── gallery                 # image gallery helper
\vert   ├── group-by-array          # group by array helper for archives
\vert   ├── nav_list                # navigation list helper
\vert   ├── toc                     # table of contents helper
\vert   └── ...
├── _layouts
\vert   ├── archive-taxonomy.html   # tag/category archive for Jekyll Archives plugin
\vert   ├── archive.html            # archive base
\vert   ├── categories.html         # archive listing posts grouped by category
\vert   ├── category.html           # archive listing posts grouped by specific category
\vert   ├── collection.html         # archive listing documents in a specific collection
\vert   ├── compress.html           # compresses HTML in pure Liquid
\vert   ├── default.html            # base for all other layouts
\vert   ├── home.html               # home page
\vert   ├── posts.html              # archive listing posts grouped by year
\vert   ├── search.html             # search page
\vert   ├── single.html             # single document (post/page/etc)
\vert   ├── tag.html                # archive listing posts grouped by specific tag
\vert   ├── tags.html               # archive listing posts grouped by tags
\vert   └── splash.html             # splash page
├── _sass                      # SCSS partials
├── assets
\vert   ├── css
\vert   \vert   └── main.scss            # main stylesheet, loads SCSS partials from _sass
\vert   ├── images                  # image assets for posts/pages/collections/etc.
\vert   ├── js
\vert   \vert   ├── plugins              # jQuery plugins
\vert   \vert   ├── vendor               # vendor scripts
\vert   \vert   ├── _main.js             # plugin settings and other scripts to load after jQuery
\vert   \vert   └── main.min.js          # optimized and concatenated script file loaded before </body>
├── _config.yml                # site configuration
├── Gemfile                    # gem file dependencies
├── index.html                 # paginated home page showing recent posts
└── package.json               # NPM build scripts
```
