import { describe, expect, it } from "vitest";
import { eq } from "drizzle-orm";
import { getDb, createPrediction } from "./db";
import { predictions } from "../drizzle/schema";
import { runAttritionInference } from "./attritionInference";

const input = {
  employeeLabel: "database-persistence-smoke-test",
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

describe("prediction persistence against the real database", () => {
  it.skipIf(!process.env.DATABASE_URL)("persists and reads a full-precision prediction row, then cleans it up", async () => {
    const db = await getDb();
    if (!db) throw new Error("DATABASE_URL is required for the persistence smoke test");

    const inference = await runAttritionInference(input);
    const selected = inference.modes?.balanced as { threshold: number; predictedAttrition: boolean };
    let predictionId: number | undefined;

    try {
      const saved = await createPrediction({
        userId: 1,
        employeeLabel: input.employeeLabel,
        mode: input.mode,
        threshold: String(selected.threshold),
        probability: String(inference.probability),
        predictedAttrition: selected.predictedAttrition ? 1 : 0,
        inputs: JSON.stringify(input),
      });
      predictionId = saved.id;

      const rows = await db.select().from(predictions).where(eq(predictions.id, predictionId)).limit(1);
      expect(rows).toHaveLength(1);
      expect(rows[0]).toMatchObject({
        userId: 1,
        employeeLabel: input.employeeLabel,
        mode: "balanced",
        threshold: "0.32",
        probability: String(inference.probability),
        inputs: JSON.stringify(input),
      });
    } finally {
      if (predictionId !== undefined) {
        await db.delete(predictions).where(eq(predictions.id, predictionId));
      }
    }
  }, 90_000);
});
