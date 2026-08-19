import { describe, expect, it } from "vitest";
import { ATTRITION_MODES, runAttritionInference } from "./attritionInference";

describe("attrition inference", () => {
  it("preserves the exact operating thresholds", () => {
    expect(ATTRITION_MODES.high_recall.threshold).toBe(0.27);
    expect(ATTRITION_MODES.balanced.threshold).toBe(0.32);
    expect(ATTRITION_MODES.high_precision.threshold).toBe(0.45);
  });

  it("loads best_f1_model.joblib and returns a bounded probability", async () => {
    const result = await runAttritionInference({
      Age: 34, BusinessTravel: "Travel_Rarely", DailyRate: 850, Department: "Research & Development", DistanceFromHome: 5,
      Education: 3, EducationField: "Life Sciences", EnvironmentSatisfaction: 3, Gender: "Female", HourlyRate: 70,
      JobInvolvement: 3, JobLevel: 2, JobRole: "Research Scientist", JobSatisfaction: 3, MaritalStatus: "Single",
      MonthlyIncome: 5000, MonthlyRate: 15000, NumCompaniesWorked: 2, OverTime: "No", PercentSalaryHike: 14,
      PerformanceRating: 3, RelationshipSatisfaction: 3, StockOptionLevel: 1, TotalWorkingYears: 8,
      TrainingTimesLastYear: 3, WorkLifeBalance: 3, YearsAtCompany: 5, YearsInCurrentRole: 3,
      YearsSinceLastPromotion: 1, YearsWithCurrManager: 3,
    });
    expect(result.modelArtifact).toBe("best_f1_model.joblib");
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
  }, 15_000);
});
