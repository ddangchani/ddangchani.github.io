export interface PostJobInput {
  topic: string;
  audience: string;
  tone: string;
  language: string;
  sourceUrls: string[];
  tagHints: string[];
  publish: boolean;
  targetBranch: string;
}

export interface PostJobResult {
  createdRoute: string | null;
  contentFile: string | null;
  assetFiles: string[];
  checksPassed: boolean;
  commitSha: string | null;
  pushed: boolean;
  failureReason: string | null;
}
