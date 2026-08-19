import { beforeEach, describe, expect, it, vi } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

const { runAttritionInference } = vi.hoisted(() => ({
  runAttritionInference: vi.fn(),
}));

vi.mock("./attritionInference", async () => {
  const actual = await vi.importActual<typeof import("./attritionInference")>("./attritionInference");
  return {
    ...actual,
    runAttritionInference,
  };
});

function context(user: TrpcContext["user"]): TrpcContext {
  return { user, req: {} as TrpcContext["req"], res: {} as TrpcContext["res"] };
}

const user = {
  id: 7,
  openId: "smoke-test-user",
  name: "Smoke Test User",
  email: "smoke@example.com",
  loginMethod: "test",
  role: "user",
  createdAt: new Date(),
  updatedAt: new Date(),
  lastSignedIn: new Date(),
} as const;

describe("predictions.smokeTest", () => {
  beforeEach(() => vi.clearAllMocks());

  it("requires authentication", async () => {
    const caller = appRouter.createCaller(context(null));
    await expect(caller.predictions.smokeTest()).rejects.toMatchObject({ code: "UNAUTHORIZED" });
  });

  it("verifies the real artifact contract for an authenticated caller", async () => {
    runAttritionInference.mockResolvedValue({
      probability: 0.18,
      modelArtifact: "best_f1_model.joblib",
      modes: {
        high_recall: { threshold: 0.27, predictedAttrition: false },
        balanced: { threshold: 0.32, predictedAttrition: false },
        high_precision: { threshold: 0.45, predictedAttrition: false },
      },
    });

    const result = await appRouter.createCaller(context(user)).predictions.smokeTest();

    expect(result.ok).toBe(true);
    expect(result.modelArtifact).toBe("best_f1_model.joblib");
    expect(result.probability).toBe(0.18);
    expect(result.modes).toHaveProperty("balanced.threshold", 0.32);
    expect(runAttritionInference).toHaveBeenCalledOnce();
    expect(runAttritionInference.mock.calls[0]?.[0]).toMatchObject({
      employeeLabel: "production-smoke-test",
      mode: "balanced",
      OverTime: "No",
    });
  });
});
