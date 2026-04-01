export const FILTER_TAGS = [
  "Statistics",
  "Machine Learning",
  "Deep Learning",
  "Python",
  "Project",
  "Time Series",
  "Gaussian Process",
  "NLP",
  "Research",
  "Paper Review",
  "Mathematics",
  "Probability Theory",
  "Bayesian",
  "Real Analysis",
  "Causal Inference",
  "Linear Model",
  "Linear Algebra",
  "Manifold Learning",
  "TDA",
  "Optimization",
  "Spatial Statistics",
  "Spectral Analysis",
  "Algorithm",
  "Jekyll",
  "Data Structure",
  "PGM",
  "Differential Privacy"
] as const;

const FILTER_TAG_SET = new Set(FILTER_TAGS.map((tag) => tag.toLocaleLowerCase()));

export function isFilterTag(tag: string): boolean {
  return FILTER_TAG_SET.has(tag.toLocaleLowerCase());
}
