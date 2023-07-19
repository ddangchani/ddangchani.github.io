---
title: "Post: Image (Caption)"
categories:
  - Post Formats
tags:
  - image
  - Post Formats
---

{% capture fig_img %}
![Foo]({{ "/assets/images/unsplash-gallery-image-3.jpg" \vert  relative_url }})
{% endcapture %}

<figure>
  {{ fig_img \vert  markdownify \vert  remove: "<p>" \vert  remove: "</p>" }}
  <figcaption>Photo from Unsplash.</figcaption>
</figure>