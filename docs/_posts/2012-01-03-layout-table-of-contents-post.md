---
title: "Layout: Post with Table of Contents"
header:
  image: assets/images/unsplash-image-9.jpg
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
tags:
  - table of contents
toc: true
toc_label: "Unique Title"
toc_icon: "heart"
---

Enable table of contents on post or page by adding `toc: true` to its YAML Front Matter. The title and icon can also be changed with:

```yaml
---
toc: true
toc_label: "Unique Title"
toc_icon: "heart"  # corresponding Font Awesome icon name (without fa prefix)
---
```

## HTML Elements

Below is just about everything you'll need to style in the theme. Check the source code to see the many embedded elements within paragraphs.

## Body text

Lorem ipsum dolor sit amet, test link adipiscing elit. **This is strong**. Nullam dignissim convallis est. Quisque aliquam.

![Smithsonian Image]({{ site.url }}{{ site.baseurl }}/assets/images/3953273590_704e3899d5_m.jpg)
{: .image-right}

*This is emphasized*. Donec faucibus. Nunc iaculis suscipit dui. 53 = 125. Water is H2O. Nam sit amet sem. Aliquam libero nisi, imperdiet at, tincidunt nec, gravida vehicula, nisl. The New York Times (That’s a citation). Underline.Maecenas ornare tortor. Donec sed tellus eget sapien fringilla nonummy. Mauris a ante. Suspendisse quam sem, consequat at, commodo vitae, feugiat in, nunc. Morbi imperdiet augue quis tellus.

HTML and CSS are our tools. Mauris a ante. Suspendisse quam sem, consequat at, commodo vitae, feugiat in, nunc. Morbi imperdiet augue quis tellus. Praesent mattis, massa quis luctus fermentum, turpis mi volutpat justo, eu volutpat enim diam eget metus.

### Blockquotes

> Lorem ipsum dolor sit amet, test link adipiscing elit. Nullam dignissim convallis est. Quisque aliquam.

## List Types

### Ordered Lists

1. Item one
   1. sub item one
   2. sub item two
   3. sub item three
2. Item two

### Unordered Lists

* Item one
* Item two
* Item three

## Tables

\vert  Header1 \vert  Header2 \vert  Header3 \vert 
\vert :--------\vert :-------:\vert --------:\vert 
\vert  cell1   \vert  cell2   \vert  cell3   \vert 
\vert  cell4   \vert  cell5   \vert  cell6   \vert 
\vert ----
\vert  cell1   \vert  cell2   \vert  cell3   \vert 
\vert  cell4   \vert  cell5   \vert  cell6   \vert 
\vert =====
\vert  Foot1   \vert  Foot2   \vert  Foot3
{: rules="groups"}

## Code Snippets

```css
#container {
  float: left;
  margin: 0 -240px 0 0;
  width: 100%;
}
```

## Buttons

Make any link standout more when applying the `.btn` class.

```html
<a href="#" class="btn btn--success">Success Button</a>
```

<div markdown="0"><a href="#" class="btn">Primary Button</a></div>
<div markdown="0"><a href="#" class="btn btn--success">Success Button</a></div>
<div markdown="0"><a href="#" class="btn btn--warning">Warning Button</a></div>
<div markdown="0"><a href="#" class="btn btn--danger">Danger Button</a></div>
<div markdown="0"><a href="#" class="btn btn--info">Info Button</a></div>

## Notices

**Watch out!** You can also add notices by appending `{: .notice}` to a paragraph.
{: .notice}