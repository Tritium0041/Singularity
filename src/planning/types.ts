export const PLAN_STEP_STATUSES = ["pending", "in_progress", "completed", "blocked", "cancelled"] as const;

export type PlanStepStatus = (typeof PLAN_STEP_STATUSES)[number];

export const PLAN_REVIEW_STATUSES = ["pending", "approved"] as const;

export type PlanReviewStatus = (typeof PLAN_REVIEW_STATUSES)[number];

export type PlanStep = {
  id: string;
  title: string;
  description?: string;
  status: PlanStepStatus;
  dependsOn?: string[];
  completionCriteria?: string[];
  evidence?: string[];
  createdAt: string;
  updatedAt: string;
};

export type PlanState = {
  objective: string;
  reviewStatus: PlanReviewStatus;
  reviewedAt?: string;
  steps: PlanStep[];
  currentStepId?: string;
  revision: number;
  createdAt: string;
  updatedAt: string;
};

export type CreatePlanStepInput = {
  id?: string;
  title: string;
  description?: string;
  dependsOn?: string[];
  completionCriteria?: string[];
  evidence?: string[];
};

export type CreatePlanInput = {
  objective: string;
  steps?: CreatePlanStepInput[];
  currentStepId?: string;
};

export type ReplacePlanInput = CreatePlanInput;

export type AddPlanStepInput = CreatePlanStepInput & {
  status?: PlanStepStatus;
  setCurrent?: boolean;
};

export type UpdatePlanStepInput = {
  id: string;
  title?: string;
  description?: string;
  status?: PlanStepStatus;
  dependsOn?: string[];
  completionCriteria?: string[];
  evidence?: string[];
  appendEvidence?: string[];
  setCurrent?: boolean;
};

export type CompletePlanStepInput = {
  id: string;
  evidence: string[];
  note?: string;
  setNextCurrent?: boolean;
};

export type BlockPlanStepInput = {
  id: string;
  reason: string;
  evidence?: string[];
};

export type ApprovePlanInput = {
  note?: string;
};
