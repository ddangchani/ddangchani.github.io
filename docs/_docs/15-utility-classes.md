---
title: "Utility Classes"
permalink: /docs/utility-classes/
excerpt: "CSS classes for aligning text/image, styling buttons and notices, and more."
last_modified_at: 2018-11-25T19:46:43-05:00
toc: true
toc_label: "Utility Classes"
toc_icon: "cogs"
---

Using the Kramdown Markdown renderer with Jekyll allows you to add [block](http://kramdown.gettalong.org/quickref.html#block-attributes) and [inline attributes](http://kramdown.gettalong.org/quickref.html#inline-attributes). This is nice if you want to add custom styling to text and image, and still write in Markdown.

**Jekyll 3:** Kramdown is the default for `jekyll new` sites and those hosted on GitHub Pages. Not using Kramdown? That's OK. The following classes are still available when used with standard HTML.
{: .notice--warning}

## Text alignment

Align text blocks with the following classes.

Left aligned text `.text-left`
{: .text-left}

```markdown
Left aligned text
{: .text-left}
```

---

Center aligned text. `.text-center`
{: .text-center}

```markdown
Center aligned text.
{: .text-center}
```

---

Right aligned text. `.text-right`
{: .text-right}

```markdown
Right aligned text.
{: .text-right}
```

---

**Justified text.** `.text-justify` Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque vel eleifend odio, eu elementum purus. In hac habitasse platea dictumst. Fusce sed sapien eleifend, sollicitudin neque non, faucibus est. Proin tempus nisi eu arcu facilisis, eget venenatis eros consequat.
{: .text-justify}

```markdown
Justified text.
{: .text-justify}
```

---

No wrap text. `.text-nowrap`
{: .text-nowrap}

```markdown
No wrap text.
{: .text-nowrap}
```

## Image alignment

Position images with the following classes.

![image-center]({{ "/assets/images/image-alignment-580x300.jpg" \vert  relative_url }}){: .align-center}

The image above happens to be **centered**.

```markdown
![image-center](/assets/images/filename.jpg){: .align-center}
```

---

![image-left]({{ "/assets/images/image-alignment-150x150.jpg" \vert  relative_url }}){: .align-left} The rest of this paragraph is filler for the sake of seeing the text wrap around the 150×150 image, which is **left aligned**. There should be plenty of room above, below, and to the right of the image. Just look at him there --- Hey guy! Way to rock that left side. I don't care what the right aligned image says, you look great. Don't let anyone else tell you differently.

```markdown
![image-left](/assets/images/filename.jpg){: .align-left}
```

---

![image-right]({{ "/assets/images/image-alignment-300x200.jpg" \vert  relative_url }}){: .align-right}

And now we're going to shift things to the **right align**. Again, there should be plenty of room above, below, and to the left of the image. Just look at him there --- Hey guy! Way to rock that right side. I don't care what the left aligned image says, you look great. Don't let anyone else tell you differently.

```markdown
![image-right](/assets/images/filename.jpg){: .align-right}
```

---

![full]({{ "/assets/images/image-alignment-1200x4002.jpg" \vert  relative_url }})
{: .full}

The image above should extend outside of the parent container on right.

```markdown
![full](/assets/images/filename.jpg)
{: .full}
```

## Buttons

Make any link standout more when applying the `.btn .btn--primary` classes.

```html
<a href="#" class="btn btn--primary">Link Text</a>
```

\vert  Button Type   \vert  Example \vert  Class \vert  Kramdown \vert 
\vert  ------        \vert  ------- \vert  ----- \vert  ------- \vert 
\vert  Default       \vert  [Text](#link){: .btn} \vert  `.btn` \vert  `[Text](#link){: .btn}` \vert 
\vert  Primary       \vert  [Text](#link){: .btn .btn--primary} \vert  `.btn .btn--primary` \vert  `[Text](#link){: .btn .btn--primary}` \vert 
\vert  Success       \vert  [Text](#link){: .btn .btn--success} \vert  `.btn .btn--success` \vert  `[Text](#link){: .btn .btn--success}` \vert 
\vert  Warning       \vert  [Text](#link){: .btn .btn--warning} \vert  `.btn .btn--warning` \vert  `[Text](#link){: .btn .btn--warning}` \vert 
\vert  Danger        \vert  [Text](#link){: .btn .btn--danger} \vert  `.btn .btn--danger` \vert  `[Text](#link){: .btn .btn--danger}` \vert 
\vert  Info          \vert  [Text](#link){: .btn .btn--info} \vert  `.btn .btn--info` \vert  `[Text](#link){: .btn .btn--info}` \vert 
\vert  Inverse       \vert  [Text](#link){: .btn .btn--inverse} \vert  `.btn .btn--inverse` \vert  `[Text](#link){: .btn .btn--inverse}` \vert 
\vert  Light Outline \vert  [Text](#link){: .btn .btn--light-outline} \vert  `.btn .btn--light-outline` \vert  `[Text](#link){: .btn .btn--light-outline}` \vert 

\vert  Button Size \vert  Example \vert  Class \vert  Kramdown \vert 
\vert  ----------- \vert  ------- \vert  ----- \vert  -------- \vert 
\vert  X-Large     \vert  [X-Large Button](#){: .btn .btn--primary .btn--x-large} \vert  `.btn .btn--primary .btn--x-large` \vert  `[Text](#link){: .btn .btn--primary .btn--x-large}` \vert 
\vert  Large       \vert  [Large Button](#){: .btn .btn--primary .btn--large} \vert  `.btn .btn--primary .btn--large` \vert  `[Text](#link){: .btn .btn--primary .btn--large}` \vert 
\vert  Default     \vert  [Default Button](#){: .btn .btn--primary} \vert  `.btn .btn--primary` \vert  `[Text](#link){: .btn .btn--primary }` \vert 
\vert  Small       \vert  [Small Button](#){: .btn .btn--primary .btn--small} \vert  `.btn .btn--primary .btn--small` \vert  `[Text](#link){: .btn .btn--primary .btn--small}` \vert 

## Notices

Call attention to a block of text.

\vert  Notice Type \vert  Class              \vert 
\vert  ----------- \vert  -----              \vert 
\vert  Default     \vert  `.notice`          \vert 
\vert  Primary     \vert  `.notice--primary` \vert 
\vert  Info        \vert  `.notice--info`    \vert 
\vert  Warning     \vert  `.notice--warning` \vert 
\vert  Success     \vert  `.notice--success` \vert 
\vert  Danger      \vert  `.notice--danger`  \vert 

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice}` class.
{: .notice}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--primary}` class.
{: .notice--primary}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--info}` class.
{: .notice--info}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--warning}` class.
{: .notice--warning}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--success}` class.
{: .notice--success}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--danger}` class.
{: .notice--danger}

{% capture notice-text %}
You can also add the `.notice` class to a `<div>` element.

* Bullet point 1
* Bullet point 2
{% endcapture %}

<div class="notice--info">
  <h4 class="no_toc">Notice Headline:</h4>
  {{ notice-text \vert  markdownify }}
</div>
