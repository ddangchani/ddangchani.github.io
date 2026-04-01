export const siteConfig = {
  title: "DDangchani's DataLog",
  description: "통계학과 데이터 사이언스를 학습하며 얻은 내용들을 공유합니다.",
  siteUrl: "https://ddangchani.github.io",
  locale: "ko-KR",
  author: {
    name: "Dangchan",
    fullName: "Dangchan Kim",
    role: "Safer society with data",
    bio: "Safer society with data",
    location: "Seoul, South Korea",
    email: "dang11230@gmail.com"
  },
  nav: [
    { href: "/posts/", label: "Articles" },
    { href: "/about/", label: "About" }
  ],
  links: [
    { href: "https://github.com/ddangchani", label: "GitHub" },
    { href: "https://www.linkedin.com/in/dangchan-kim-13932b213/", label: "LinkedIn" },
    { href: "mailto:dang11230@gmail.com", label: "Email" }
  ],
  utterances: {
    repo: "ddangchani/ddangchani.github.io",
    issueTerm: "pathname",
    label: "commentary"
  },
  highlights: [
    { label: "Topics", value: "Statistics · Data Science" },
    { label: "Writing", value: "공부한 내용을 정리해 공유" },
    { label: "Goal", value: "Safer society with data" }
  ]
} as const;
