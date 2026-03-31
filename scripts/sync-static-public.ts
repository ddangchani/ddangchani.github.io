import fs from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const publicDir = path.join(root, "public");
const assetSource = path.join(root, "assets");
const assetTarget = path.join(publicDir, "assets");

const rootFiles = [
  "ads.txt",
  "google7fbd52212214a748.html",
  "googled4154600d2f2b979.html",
  "naver9511e6c4477b457015b9955e589ea84c.html",
  "robots.txt"
];

async function ensureCleanDirectory(targetPath: string) {
  await fs.rm(targetPath, { recursive: true, force: true });
  await fs.mkdir(targetPath, { recursive: true });
}

async function copyRootFiles() {
  for (const fileName of rootFiles) {
    await fs.copyFile(path.join(root, fileName), path.join(publicDir, fileName));
  }
}

async function main() {
  await fs.mkdir(publicDir, { recursive: true });
  await copyRootFiles();
  await ensureCleanDirectory(assetTarget);
  await fs.cp(assetSource, assetTarget, { recursive: true, force: true });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
