import { ATTRITION_MODES, runAttritionInference } from "./attritionInference";

const SMOKE_PAYLOAD: Record<string, unknown> = {
  employeeLabel: "production-smoke-test",
  mode: "balanced",
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

export async function runAuthenticatedAttritionSmokeTest() {
  const result = await runAttritionInference(SMOKE_PAYLOAD);
  const modeResults = result.modes as Record<string, { threshold: number; predictedAttrition: boolean }> | undefined;
  const expectedModes = Object.keys(ATTRITION_MODES);
  const missingModes = expectedModes.filter(mode => !modeResults?.[mode]);

  if (typeof result.probability !== "number" || result.probability < 0 || result.probability > 1) {
    throw new Error("Smoke test returned an invalid probability");
  }
  if (result.modelArtifact !== "best_f1_model.joblib") {
    throw new Error(`Smoke test loaded an unexpected artifact: ${result.modelArtifact ?? "unknown"}`);
  }
  if (missingModes.length > 0) {
    throw new Error(`Smoke test response is missing operating modes: ${missingModes.join(", ")}`);
  }

  return {
    ok: true as const,
    modelArtifact: result.modelArtifact,
    probability: result.probability,
    modes: modeResults,
    checkedAt: new Date().toISOString(),
  };
}
