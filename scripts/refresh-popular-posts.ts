import fs from "node:fs";
import path from "node:path";

import { BetaAnalyticsDataClient } from "@google-analytics/data";

const DEFAULT_PROPERTY_ID = "397192433";
const EXCLUDED_ROUTES = new Set(["/", "/about/", "/posts/", "/tags/", "/dsroadmap/"]);

function fail(message: string): never {
  throw new Error(message);
}

function getCredentials() {
  const raw = process.env.GOOGLE_ANALYTICS_CREDENTIALS_JSON;

  if (!raw) {
    fail("GOOGLE_ANALYTICS_CREDENTIALS_JSON is required");
  }

  return JSON.parse(raw) as {
    client_email: string;
    private_key: string;
  };
}

function getClient(): BetaAnalyticsDataClient {
  return new BetaAnalyticsDataClient({
    credentials: getCredentials()
  });
}

function getDateRangeStart(daysBack: number): string {
  const target = new Date();
  target.setDate(target.getDate() - daysBack);
  return target.toISOString().slice(0, 10);
}

function getYesterday(): string {
  const target = new Date();
  target.setDate(target.getDate() - 1);
  return target.toISOString().slice(0, 10);
}

async function runReport(startDate: string, outputPath: string) {
  const client = getClient();
  const propertyId = process.env.GA_PROPERTY_ID ?? DEFAULT_PROPERTY_ID;
  const [response] = await client.runReport({
    property: `properties/${propertyId}`,
    dimensions: [{ name: "pagePath" }],
    metrics: [{ name: "screenPageViews" }],
    dateRanges: [{ startDate, endDate: getYesterday() }]
  });

  const analytics: Record<string, { link: string; count: number }> = {};

  for (const row of response.rows ?? []) {
    const route = row.dimensionValues?.[0]?.value ?? "";
    const count = Number(row.metricValues?.[0]?.value ?? "0");
    const decodedRoute = decodeURIComponent(route);

    if (!route || EXCLUDED_ROUTES.has(decodedRoute)) {
      continue;
    }

    analytics[route] = {
      link: route,
      count
    };
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(analytics, null, 2)}\n`, "utf8");
}

async function main() {
  await runReport("2023-01-01", path.join("_data", "analytics.json"));
  await runReport(getDateRangeStart(30), path.join("_data", "analytics_month.json"));
  console.log("Popularity data refreshed");
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
