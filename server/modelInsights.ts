import fs from "node:fs";
import path from "node:path";

const insightRoot = path.resolve(process.cwd(), "ml_artifacts/insights");

function readCsv(fileName: string) {
  const file = path.join(insightRoot, fileName);
  if (!fs.existsSync(file)) return [] as Record<string, string>[];
  const [headerLine, ...rows] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return rows.filter(Boolean).map(row => {
    const values = row.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

export function getModelInsights() {
  const featureImportance = readCsv("permutation_importance.csv")
    .slice(0, 8)
    .map(row => ({ feature: row.feature, value: Number(row.importance_mean || 0) }));
  const frontier = readCsv("constrained_frontier_dev.csv")
    .filter(row => row.threshold)
    .slice(0, 8)
    .map(row => ({
      constraint: row.constraint,
      threshold: Number(row.threshold),
      precision: Number(row.precision),
      recall: Number(row.recall),
      f1: Number(row.f1),
    }));
  return { featureImportance, frontier };
}
