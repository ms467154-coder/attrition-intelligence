import { beforeEach, describe, expect, it, vi } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

const { createPrediction, getPredictionsByUser, runAttritionInference } = vi.hoisted(() => ({
  createPrediction: vi.fn(),
  getPredictionsByUser: vi.fn(),
  runAttritionInference: vi.fn(),
}));

vi.mock("./db", () => ({ createPrediction, getPredictionsByUser }));
vi.mock("./attritionInference", () => ({
  ATTRITION_MODES: {
    high_recall: { label: "High Recall", threshold: 0.27 },
    balanced: { label: "Balanced", threshold: 0.32 },
    high_precision: { label: "High Precision", threshold: 0.45 },
  },
  runAttritionInference,
}));

const input = {
  employeeLabel: "QA Employee", mode: "balanced" as const, Age: 34, BusinessTravel: "Travel_Rarely", DailyRate: 850,
  Department: "Research & Development", DistanceFromHome: 5, Education: 3, EducationField: "Life Sciences",
  EnvironmentSatisfaction: 3, Gender: "Female", HourlyRate: 70, JobInvolvement: 3, JobLevel: 2,
  JobRole: "Research Scientist", JobSatisfaction: 3, MaritalStatus: "Single", MonthlyIncome: 5000,
  MonthlyRate: 15000, NumCompaniesWorked: 2, OverTime: "No", PercentSalaryHike: 14, PerformanceRating: 3,
  RelationshipSatisfaction: 3, StockOptionLevel: 1, TotalWorkingYears: 8, TrainingTimesLastYear: 3,
  WorkLifeBalance: 3, YearsAtCompany: 5, YearsInCurrentRole: 3, YearsSinceLastPromotion: 1, YearsWithCurrManager: 3,
};

function context(user: TrpcContext["user"]): TrpcContext {
  return { user, req: {} as TrpcContext["req"], res: {} as TrpcContext["res"] };
}

describe("predictions router", () => {
  beforeEach(() => vi.clearAllMocks());

  it("requires authentication for history", async () => {
    const caller = appRouter.createCaller(context(null));
    await expect(caller.predictions.history()).rejects.toMatchObject({ code: "UNAUTHORIZED" });
  });

  it("runs inference and persists a validated prediction payload", async () => {
    runAttritionInference.mockResolvedValue({ probability: 0.41, modelArtifact: "best_f1_model.joblib", modes: { balanced: { threshold: 0.32, predictedAttrition: true } } });
    createPrediction.mockResolvedValue({ id: 42 });
    const user = { id: 7, openId: "test-user", name: "Test User", email: "test@example.com", loginMethod: "test", role: "user", createdAt: new Date(), updatedAt: new Date(), lastSignedIn: new Date() } as const;
    const result = await appRouter.createCaller(context(user)).predictions.create(input);
    expect(result.predictionId).toBe(42);
    expect(createPrediction).toHaveBeenCalledWith(expect.objectContaining({ userId: 7, mode: "balanced", threshold: "0.32", probability: "0.41", predictedAttrition: 1 }));
  });

  it("returns only the authenticated user's history", async () => {
    getPredictionsByUser.mockResolvedValue([{ id: 1, userId: 7, employeeLabel: "Employee A" }]);
    const user = { id: 7, openId: "test-user", name: "Test User", email: "test@example.com", loginMethod: "test", role: "user", createdAt: new Date(), updatedAt: new Date(), lastSignedIn: new Date() } as const;
    const result = await appRouter.createCaller(context(user)).predictions.history();
    expect(getPredictionsByUser).toHaveBeenCalledWith(7);
    expect(result).toHaveLength(1);
  });
});
