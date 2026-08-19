import { describe, expect, it } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

function context(user: TrpcContext["user"]): TrpcContext {
  return { user, req: {} as TrpcContext["req"], res: {} as TrpcContext["res"] };
}

const user = {
  id: 991,
  openId: "real-smoke-test-user",
  name: "Real Smoke Test User",
  email: "real-smoke@example.com",
  loginMethod: "test",
  role: "user",
  createdAt: new Date(),
  updatedAt: new Date(),
  lastSignedIn: new Date(),
} as const;

describe("authenticated ML smoke test integration", () => {
  it("loads best_f1_model.joblib and returns valid operating modes", async () => {
    const result = await appRouter.createCaller(context(user)).predictions.smokeTest();

    expect(result.ok).toBe(true);
    expect(result.modelArtifact).toBe("best_f1_model.joblib");
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
    expect(result.modes.high_recall.threshold).toBe(0.27);
    expect(result.modes.balanced.threshold).toBe(0.32);
    expect(result.modes.high_precision.threshold).toBe(0.45);
    expect(result.checkedAt).toMatch(/Z$/);
  }, 90_000);
});
