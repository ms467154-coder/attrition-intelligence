import { z } from "zod";
import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { protectedProcedure, publicProcedure, router } from "./_core/trpc";
import { createPrediction, getPredictionsByUser } from "./db";
import { ATTRITION_MODES, runAttritionInference } from "./attritionInference";
import { getModelInsights } from "./modelInsights";
import { runAuthenticatedAttritionSmokeTest } from "./attritionSmokeTest";

const employeeInput = z.object({
  employeeLabel: z.string().trim().min(1).max(120),
  mode: z.enum(["high_recall", "balanced", "high_precision"]),
  Age: z.coerce.number().int().min(18).max(80),
  BusinessTravel: z.string().min(1), DailyRate: z.coerce.number().int().min(0), Department: z.string().min(1),
  DistanceFromHome: z.coerce.number().int().min(0), Education: z.coerce.number().int().min(1).max(5), EducationField: z.string().min(1),
  EnvironmentSatisfaction: z.coerce.number().int().min(1).max(4), Gender: z.string().min(1), HourlyRate: z.coerce.number().int().min(0),
  JobInvolvement: z.coerce.number().int().min(1).max(4), JobLevel: z.coerce.number().int().min(1).max(5), JobRole: z.string().min(1),
  JobSatisfaction: z.coerce.number().int().min(1).max(4), MaritalStatus: z.string().min(1), MonthlyIncome: z.coerce.number().int().min(0),
  MonthlyRate: z.coerce.number().int().min(0), NumCompaniesWorked: z.coerce.number().int().min(0), OverTime: z.string().min(1),
  PercentSalaryHike: z.coerce.number().int().min(0), PerformanceRating: z.coerce.number().int().min(1).max(5), RelationshipSatisfaction: z.coerce.number().int().min(1).max(4),
  StockOptionLevel: z.coerce.number().int().min(0).max(3), TotalWorkingYears: z.coerce.number().int().min(0), TrainingTimesLastYear: z.coerce.number().int().min(0),
  WorkLifeBalance: z.coerce.number().int().min(1).max(4), YearsAtCompany: z.coerce.number().int().min(0), YearsInCurrentRole: z.coerce.number().int().min(0),
  YearsSinceLastPromotion: z.coerce.number().int().min(0), YearsWithCurrManager: z.coerce.number().int().min(0),
});

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return { success: true } as const;
    }),
  }),
  metrics: publicProcedure.query(() => ({
    cvF1: 0.6536, testF1: 0.5882, prAuc: 0.6086,
    model: "Logistic Regression / XGBoost ensemble",
    artifact: "best_f1_model.joblib",
    modes: Object.entries(ATTRITION_MODES).map(([key, value]) => ({ key, ...value })),
    insights: getModelInsights(),
  })),
  predictions: router({
    history: protectedProcedure.query(({ ctx }) => getPredictionsByUser(ctx.user.id)),
    smokeTest: protectedProcedure.query(() => runAuthenticatedAttritionSmokeTest()),
    create: protectedProcedure.input(employeeInput).mutation(async ({ ctx, input }) => {
      const result = await runAttritionInference(input);
      const selected = result.modes?.[input.mode] as { threshold: number; predictedAttrition: boolean };
      const saved = await createPrediction({
        userId: ctx.user.id,
        employeeLabel: input.employeeLabel,
        mode: input.mode,
        threshold: String(selected.threshold),
        probability: String(result.probability),
        predictedAttrition: selected.predictedAttrition ? 1 : 0,
        inputs: JSON.stringify(input),
      });
      return { ...result, selectedMode: input.mode, selected, predictionId: saved.id };
    }),
  }),
});

export type AppRouter = typeof appRouter;
