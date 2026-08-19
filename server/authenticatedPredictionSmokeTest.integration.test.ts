import { describe, expect, it, vi } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

const { createPrediction } = vi.hoisted(() => ({
  createPrediction: vi.fn(),
}));

vi.mock("./db", async () => {
  const actual = await vi.importActual<typeof import("./db")>("./db");
  return { ...actual, createPrediction };
});

function context(user: TrpcContext["user"]): TrpcContext {
  return { user, req: {} as TrpcContext["req"], res: {} as TrpcContext["res"] };
}

const user = {
  id: 992,
  openId: "prediction-smoke-test-user",
  name: "Prediction Smoke Test User",
  email: "prediction-smoke@example.com",
  loginMethod: "test",
  role: "user",
  createdAt: new Date(),
  updatedAt: new Date(),
  lastSignedIn: new Date(),
} as const;

const input = {
  employeeLabel: "production-prediction-smoke-test",
  mode: "balanced" as const,
  Age: 34,
  BusinessTravel: "Travel_Rarely",
  DailyRate: 850,
  Department: "Research & Development",
  DistanceFromHome: 5,
  Education: 3,
  EducationField: "Life Sciences",
  EnvironmentSatisfaction: 3,
  Gender: "Female",
  HourlyRate: 70,
  JobInvolvement: 3,
  JobLevel: 2,
  JobRole: "Research Scientist",
  JobSatisfaction: 3,
  MaritalStatus: "Single",
  MonthlyIncome: 5000,
  MonthlyRate: 15000,
  NumCompaniesWorked: 2,
  OverTime: "No",
  PercentSalaryHike: 14,
  PerformanceRating: 3,
  RelationshipSatisfaction: 3,
  StockOptionLevel: 1,
  TotalWorkingYears: 8,
  TrainingTimesLastYear: 3,
  WorkLifeBalance: 3,
  YearsAtCompany: 5,
  YearsInCurrentRole: 3,
  YearsSinceLastPromotion: 1,
  YearsWithCurrManager: 3,
};

describe("authenticated prediction creation smoke test", () => {
  it("loads the real model and completes the protected create path", async () => {
    createPrediction.mockResolvedValue({ id: 993 });

    const result = await appRouter.createCaller(context(user)).predictions.create(input);

    expect(result.predictionId).toBe(993);
    expect(result.modelArtifact).toBe("best_f1_model.joblib");
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
    expect(result.selectedMode).toBe("balanced");
    expect(result.selected.threshold).toBe(0.32);
    expect(createPrediction).toHaveBeenCalledWith(expect.objectContaining({
      userId: 992,
      employeeLabel: "production-prediction-smoke-test",
      mode: "balanced",
      threshold: "0.32",
    }));
  }, 90_000);
});
