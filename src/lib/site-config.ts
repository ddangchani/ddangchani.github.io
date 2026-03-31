export const siteConfig = {
  title: "DDangchani's DataLog",
  description: "A technical writing archive focused on data science, AI, statistics, and practical research notes.",
  siteUrl: "https://ddangchani.github.io",
  locale: "ko-KR",
  author: {
    name: "Dangchan",
    role: "Data scientist and technical writer",
    bio: "Safer society with data. Research notes, implementation details, and long-form technical study.",
    location: "South Korea",
    email: "dang11230@gmail.com"
  },
  nav: [
    { href: "/", label: "Index" },
    { href: "/posts/", label: "Archive" },
    { href: "/search/", label: "Search" },
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
    { label: "Topics", value: "AI · Statistics · Data" },
    { label: "Format", value: "Long-form research notes" },
    { label: "Mode", value: "Static React archive" }
  ]
} as const;
