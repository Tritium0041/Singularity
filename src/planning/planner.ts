import {
  PLAN_STEP_STATUSES,
  PLAN_REVIEW_STATUSES,
  type AddPlanStepInput,
  type ApprovePlanInput,
  type BlockPlanStepInput,
  type CompletePlanStepInput,
  type CreatePlanInput,
  type CreatePlanStepInput,
  type PlanState,
  type PlanReviewStatus,
  type PlanStep,
  type PlanStepStatus,
  type ReplacePlanInput,
  type UpdatePlanStepInput
} from "./types.js";

export class Planner {
  private plan: PlanState | undefined;

  constructor(initial?: PlanState) {
    this.plan = initial === undefined ? undefined : validateAndCloneState(initial);
  }

  get state(): PlanState | undefined {
    return this.plan === undefined ? undefined : cloneState(this.plan);
  }

  create(input: CreatePlanInput): PlanState {
    if (this.plan) {
      throw new Error("Plan already exists. Use replace to overwrite it.");
    }
    this.plan = buildPlan(input);
    return cloneState(this.plan);
  }

  replace(input: ReplacePlanInput): PlanState {
    this.plan = buildPlan(input);
    return cloneState(this.plan);
  }

  approve(input: ApprovePlanInput = {}): PlanState {
    const plan = this.requirePlan();
    normalizeOptionalString(input.note);
    if (plan.reviewStatus !== "approved") {
      plan.reviewStatus = "approved";
      plan.reviewedAt = new Date().toISOString();
      touch(plan);
    }
    return cloneState(plan);
  }

  addStep(input: AddPlanStepInput): PlanStep {
    const plan = this.requirePlan();
    const now = new Date().toISOString();
    const step = buildStep(input, now, input.status ?? "pending");
    ensureUniqueStepId(plan.steps, step.id);
    ensureDependenciesExist([...plan.steps, step], step.dependsOn);
    if (step.status === "in_progress") {
      ensureDependenciesCompleted(plan, step);
    }
    if (step.status === "completed") {
      ensureCompletionEvidence(step.evidence);
    }
    plan.steps.push(step);
    markPlanReviewPending(plan);
    if (input.setCurrent || !plan.currentStepId) {
      ensureCanBeCurrent(plan, step);
      plan.currentStepId = step.id;
    }
    touch(plan);
    return cloneStep(step);
  }

  updateStep(input: UpdatePlanStepInput): PlanStep {
    const plan = this.requirePlan();
    const step = findStep(plan, input.id);
    const nextStatus = input.status ?? step.status;

    if (input.title !== undefined) {
      step.title = normalizeRequiredString(input.title, "Plan step title");
    }
    if (input.description !== undefined) {
      step.description = normalizeOptionalString(input.description);
    }
    if (input.dependsOn !== undefined) {
      step.dependsOn = normalizeStringList(input.dependsOn, "Plan step dependencies");
      ensureDependenciesExist(plan.steps, step.dependsOn);
    }
    if (input.completionCriteria !== undefined) {
      step.completionCriteria = normalizeStringList(input.completionCriteria, "Plan step completion criteria");
    }
    if (input.evidence !== undefined) {
      step.evidence = normalizeStringList(input.evidence, "Plan step evidence");
    }
    if (input.appendEvidence !== undefined) {
      step.evidence = [...(step.evidence ?? []), ...normalizeStringList(input.appendEvidence, "Plan step evidence")];
    }
    validateStatus(nextStatus);
    if (nextStatus === "in_progress") {
      ensureDependenciesCompleted(plan, step);
    }
    if (nextStatus === "completed") {
      ensureCompletionEvidence(step.evidence);
    }
    step.status = nextStatus;
    step.updatedAt = new Date().toISOString();
    if (shouldInvalidatePlanReview(input)) {
      markPlanReviewPending(plan);
    }

    if (input.setCurrent) {
      ensureCanBeCurrent(plan, step);
      plan.currentStepId = step.id;
    } else if (plan.currentStepId === step.id && (step.status === "cancelled" || step.status === "completed")) {
      plan.currentStepId = findNextCurrentStep(plan)?.id;
    }

    touch(plan);
    return cloneStep(step);
  }

  setCurrentStep(id: string): PlanStep {
    const plan = this.requirePlan();
    const step = findStep(plan, id);
    ensureCanBeCurrent(plan, step);
    if (step.status === "pending") {
      ensureDependenciesCompleted(plan, step);
      step.status = "in_progress";
      step.updatedAt = new Date().toISOString();
    }
    plan.currentStepId = step.id;
    touch(plan);
    return cloneStep(step);
  }

  completeStep(input: CompletePlanStepInput): PlanStep {
    const evidence = normalizeStringList(input.evidence, "Plan step evidence");
    const note = normalizeOptionalString(input.note);
    const step = this.updateStep({
      id: input.id,
      status: "completed",
      appendEvidence: note ? [...evidence, note] : evidence
    });
    if (input.setNextCurrent) {
      const plan = this.requirePlan();
      plan.currentStepId = findNextCurrentStep(plan)?.id;
      touch(plan);
    }
    return step;
  }

  blockStep(input: BlockPlanStepInput): PlanStep {
    const evidence = normalizeStringList(input.evidence ?? [], "Plan step evidence");
    const reason = normalizeRequiredString(input.reason, "Plan block reason");
    return this.updateStep({
      id: input.id,
      status: "blocked",
      appendEvidence: [reason, ...evidence]
    });
  }

  clear(): void {
    this.plan = undefined;
  }

  private requirePlan(): PlanState {
    if (!this.plan) {
      throw new Error("No active plan. Create a plan first.");
    }
    return this.plan;
  }
}

export function hasOpenPlanSteps(state: PlanState | undefined): boolean {
  return Boolean(state?.steps.some((step) => step.status === "pending" || step.status === "in_progress"));
}

function buildPlan(input: CreatePlanInput): PlanState {
  const now = new Date().toISOString();
  const steps = (input.steps ?? []).map((step) => buildStep(step, now, "pending"));
  const ids = new Set<string>();
  for (const step of steps) {
    if (ids.has(step.id)) {
      throw new Error(`Duplicate plan step id: ${step.id}`);
    }
    ids.add(step.id);
  }
  for (const step of steps) {
    ensureDependenciesExist(steps, step.dependsOn);
  }

  const plan: PlanState = {
    objective: normalizeRequiredString(input.objective, "Plan objective"),
    reviewStatus: "pending",
    steps,
    revision: 1,
    createdAt: now,
    updatedAt: now
  };
  const currentStepId = input.currentStepId ?? steps.find((step) => step.status !== "cancelled" && step.status !== "completed")?.id;
  if (currentStepId) {
    const current = findStep(plan, currentStepId);
    ensureCanBeCurrent(plan, current);
    plan.currentStepId = currentStepId;
  }
  return plan;
}

function buildStep(input: CreatePlanStepInput, now: string, status: PlanStepStatus): PlanStep {
  validateStatus(status);
  const step: PlanStep = {
    id: normalizeStepId(input.id),
    title: normalizeRequiredString(input.title, "Plan step title"),
    status,
    createdAt: now,
    updatedAt: now
  };
  const description = normalizeOptionalString(input.description);
  if (description !== undefined) {
    step.description = description;
  }
  const dependsOn = normalizeStringList(input.dependsOn ?? [], "Plan step dependencies");
  if (dependsOn.length > 0) {
    step.dependsOn = dependsOn;
  }
  const completionCriteria = normalizeStringList(input.completionCriteria ?? [], "Plan step completion criteria");
  if (completionCriteria.length > 0) {
    step.completionCriteria = completionCriteria;
  }
  const evidence = normalizeStringList(input.evidence ?? [], "Plan step evidence");
  if (evidence.length > 0) {
    step.evidence = evidence;
  }
  return step;
}

function validateAndCloneState(state: PlanState): PlanState {
  const objective = normalizeRequiredString(state.objective, "Plan objective");
  if (!Number.isInteger(state.revision) || state.revision < 1) {
    throw new Error("Plan revision must be a positive integer.");
  }
  const reviewStatus = normalizeReviewStatus(state.reviewStatus ?? "approved");
  const steps = state.steps.map((step) => {
    validateStatus(step.status);
    const cloned: PlanStep = {
      id: normalizeRequiredString(step.id, "Plan step id"),
      title: normalizeRequiredString(step.title, "Plan step title"),
      status: step.status,
      createdAt: normalizeRequiredString(step.createdAt, "Plan step createdAt"),
      updatedAt: normalizeRequiredString(step.updatedAt, "Plan step updatedAt")
    };
    const description = normalizeOptionalString(step.description);
    if (description !== undefined) {
      cloned.description = description;
    }
    const dependsOn = normalizeStringList(step.dependsOn ?? [], "Plan step dependencies");
    if (dependsOn.length > 0) {
      cloned.dependsOn = dependsOn;
    }
    const completionCriteria = normalizeStringList(step.completionCriteria ?? [], "Plan step completion criteria");
    if (completionCriteria.length > 0) {
      cloned.completionCriteria = completionCriteria;
    }
    const evidence = normalizeStringList(step.evidence ?? [], "Plan step evidence");
    if (evidence.length > 0) {
      cloned.evidence = evidence;
    }
    if (cloned.status === "completed") {
      ensureCompletionEvidence(cloned.evidence);
    }
    return cloned;
  });
  const ids = new Set<string>();
  for (const step of steps) {
    if (ids.has(step.id)) {
      throw new Error(`Duplicate plan step id: ${step.id}`);
    }
    ids.add(step.id);
  }
  for (const step of steps) {
    ensureDependenciesExist(steps, step.dependsOn);
  }
  const plan: PlanState = {
    objective,
    reviewStatus,
    steps,
    revision: state.revision,
    createdAt: normalizeRequiredString(state.createdAt, "Plan createdAt"),
    updatedAt: normalizeRequiredString(state.updatedAt, "Plan updatedAt")
  };
  const reviewedAt = normalizeOptionalString(state.reviewedAt);
  if (reviewedAt !== undefined) {
    plan.reviewedAt = reviewedAt;
  }
  if (plan.reviewStatus === "approved" && plan.reviewedAt === undefined) {
    plan.reviewedAt = plan.updatedAt;
  }
  if (state.currentStepId !== undefined) {
    const current = findStep(plan, state.currentStepId);
    ensureCanBeCurrent(plan, current);
    plan.currentStepId = current.id;
  }
  return plan;
}

function touch(plan: PlanState): void {
  plan.revision += 1;
  plan.updatedAt = new Date().toISOString();
}

function markPlanReviewPending(plan: PlanState): void {
  plan.reviewStatus = "pending";
  delete plan.reviewedAt;
}

function shouldInvalidatePlanReview(input: UpdatePlanStepInput): boolean {
  if (
    input.title !== undefined ||
    input.description !== undefined ||
    input.dependsOn !== undefined ||
    input.completionCriteria !== undefined
  ) {
    return true;
  }
  return false;
}

function findStep(plan: PlanState, id: string): PlanStep {
  const normalizedId = normalizeRequiredString(id, "Plan step id");
  const step = plan.steps.find((candidate) => candidate.id === normalizedId);
  if (!step) {
    throw new Error(`Plan step not found: ${normalizedId}`);
  }
  return step;
}

function findNextCurrentStep(plan: PlanState): PlanStep | undefined {
  return plan.steps.find((step) => step.status === "in_progress") ?? plan.steps.find((step) => step.status === "pending");
}

function ensureCanBeCurrent(plan: PlanState, step: PlanStep): void {
  if (step.status === "cancelled" || step.status === "completed") {
    throw new Error(`Plan step ${step.id} cannot be current because it is ${step.status}.`);
  }
  if (step.status === "pending" || step.status === "in_progress") {
    ensureDependenciesCompleted(plan, step);
  }
}

function ensureDependenciesCompleted(plan: PlanState, step: PlanStep): void {
  for (const dependencyId of step.dependsOn ?? []) {
    const dependency = findStep(plan, dependencyId);
    if (dependency.status !== "completed") {
      throw new Error(`Plan step ${step.id} depends on incomplete step ${dependencyId}.`);
    }
  }
}

function ensureDependenciesExist(steps: PlanStep[], dependsOn: string[] | undefined): void {
  for (const dependencyId of dependsOn ?? []) {
    if (!steps.some((step) => step.id === dependencyId)) {
      throw new Error(`Plan dependency does not exist: ${dependencyId}`);
    }
  }
}

function ensureUniqueStepId(steps: PlanStep[], id: string): void {
  if (steps.some((step) => step.id === id)) {
    throw new Error(`Duplicate plan step id: ${id}`);
  }
}

function ensureCompletionEvidence(evidence: string[] | undefined): void {
  if (!evidence || evidence.length === 0) {
    throw new Error("Completed plan steps must include evidence.");
  }
}

function validateStatus(status: string): asserts status is PlanStepStatus {
  if (!(PLAN_STEP_STATUSES as readonly string[]).includes(status)) {
    throw new Error(`Invalid plan step status: ${status}`);
  }
}

function normalizeReviewStatus(status: string): PlanReviewStatus {
  if (!(PLAN_REVIEW_STATUSES as readonly string[]).includes(status)) {
    throw new Error(`Invalid plan review status: ${status}`);
  }
  return status as PlanReviewStatus;
}

function normalizeStepId(id: string | undefined): string {
  return normalizeRequiredString(id ?? createPlanStepId(), "Plan step id");
}

function normalizeRequiredString(value: string, label: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${label} must be non-empty.`);
  }
  return value.trim();
}

function normalizeOptionalString(value: string | undefined): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  const normalized = value.trim();
  return normalized === "" ? undefined : normalized;
}

function normalizeStringList(values: readonly string[], label: string): string[] {
  if (!Array.isArray(values)) {
    throw new Error(`${label} must be an array of strings.`);
  }
  const normalized = values.map((value) => {
    if (typeof value !== "string") {
      throw new Error(`${label} must be an array of strings.`);
    }
    return value.trim();
  }).filter(Boolean);
  return [...new Set(normalized)];
}

function cloneState(state: PlanState): PlanState {
  return {
    ...state,
    steps: state.steps.map(cloneStep)
  };
}

function cloneStep(step: PlanStep): PlanStep {
  return {
    ...step,
    dependsOn: step.dependsOn === undefined ? undefined : [...step.dependsOn],
    completionCriteria: step.completionCriteria === undefined ? undefined : [...step.completionCriteria],
    evidence: step.evidence === undefined ? undefined : [...step.evidence]
  };
}

function createPlanStepId(): string {
  return `step_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
