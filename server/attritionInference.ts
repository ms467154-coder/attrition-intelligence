import { spawn } from "node:child_process";
import path from "node:path";

export const ATTRITION_MODES = {
  high_recall: { label: "High Recall", threshold: 0.27 },
  balanced: { label: "Balanced", threshold: 0.32 },
  high_precision: { label: "High Precision", threshold: 0.45 },
} as const;

export type AttritionMode = keyof typeof ATTRITION_MODES;

export async function runAttritionInference(input: Record<string, unknown>) {
  const projectRoot = process.env.APP_ROOT || process.cwd();
  const script = path.resolve(projectRoot, "server/ml/infer.py");
  const python = process.env.PYTHON_BIN || "/usr/bin/python3";

  const output = await new Promise<string>((resolve, reject) => {
    const child = spawn(python, [script], {
      cwd: projectRoot,
      env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONPATH: path.resolve(projectRoot, "server/ml") },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", chunk => { stdout += chunk.toString(); });
    child.stderr.on("data", chunk => { stderr += chunk.toString(); });
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("ML inference timed out"));
    }, 90_000);
    child.on("error", error => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", code => {
      clearTimeout(timer);
      if (code !== 0) {
        const detail = stderr.trim() || stdout.trim() || `process exited with code ${code}`;
        console.error(`[ML inference] ${detail}`);
        reject(new Error(`ML inference failed: ${detail}`));
      } else resolve(stdout.trim());
    });
    child.stdin.end(JSON.stringify(input));
  });

  const result = JSON.parse(output) as {
    probability?: number;
    modes?: Record<string, unknown>;
    modelArtifact?: string;
    error?: string;
  };
  if (result.error || typeof result.probability !== "number") {
    throw new Error(result.error || "The ML inference runner returned an invalid response");
  }
  return result;
}
