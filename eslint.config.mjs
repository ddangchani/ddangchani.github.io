import nextVitals from "eslint-config-next/core-web-vitals";

export default [
  {
    ignores: [
      ".next/**",
      "_posts/**",
      "assets/**",
      "content/**",
      "generated/**",
      "public/**",
      "scripts/**"
    ]
  },
  ...nextVitals
];
