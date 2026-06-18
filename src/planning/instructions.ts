import type { PromptFragment } from "../context/types.js";
import type { PlanState } from "./types.js";

export const PLANNING_INSTRUCTIONS_FRAGMENT_ID = "planning-instructions";

export function buildPlanningInstructions(options: { requirePlanBeforeMutation?: boolean } = {}): PromptFragment {
  const lines = [
    "Planning tools are available for multi-step work.",
    "- For complex tasks, create a structured plan before executing the work.",
    "- If the user request contains an explicit planning directive, you must enter plan mode and call create_plan before any execution.",
    "- After create_plan succeeds, send the plan details to the user and ask them to review/approve it before execution.",
    "- Do not call approve_plan yourself unless the user has approved or revised the pending plan.",
    "- While the current plan reviewStatus is pending, do not execute write/execute tools or begin real task execution.",
    "- Use read_plan to inspect current progress instead of relying only on chat history.",
    "- After completing a real subtask, update the related plan step with evidence.",
    "- If the plan is wrong or blocked, update the plan before continuing.",
    "- When a plan has any pending or in_progress steps, do not produce a final answer or end the conversation.",
    "- Only produce the final answer after every plan step has left pending and in_progress status.",
    "- Before the final answer, confirm required steps are completed or clearly report blockers."
  ];
  if (options.requirePlanBeforeMutation !== false) {
    lines.push(
      "- When no plan exists yet, only gather information with read-only tools and then call create_plan.",
      "- Before create_plan succeeds, do not write files, append files, execute shell commands, or write memory/workspace notes."
    );
  }

  return {
    id: PLANNING_INSTRUCTIONS_FRAGMENT_ID,
    stable: true,
    content: lines.join("\n")
  };
}

export function formatPlanSnapshot(state: PlanState | undefined, options: { maxSteps?: number } = {}): string | undefined {
  if (!state) {
    return undefined;
  }
  const maxSteps = clampMaxSteps(options.maxSteps ?? 20);
  const visibleSteps = state.steps.slice(0, maxSteps);
  const lines = [
    `<current_plan revision="${escapeXml(String(state.revision))}" review_status="${escapeXml(state.reviewStatus)}">`,
    `  <objective>${escapeXml(state.objective)}</objective>`
  ];
  if (state.reviewedAt) {
    lines.push(`  <reviewed_at>${escapeXml(state.reviewedAt)}</reviewed_at>`);
  }
  if (state.currentStepId) {
    lines.push(`  <current_step id="${escapeXml(state.currentStepId)}" />`);
  }
  for (const step of visibleSteps) {
    const current = step.id === state.currentStepId ? ` current="true"` : "";
    lines.push(`  <step id="${escapeXml(step.id)}" status="${escapeXml(step.status)}"${current}>${escapeXml(step.title)}</step>`);
  }
  if (state.steps.length > visibleSteps.length) {
    lines.push(`  <truncated_steps remaining="${state.steps.length - visibleSteps.length}" />`);
  }
  lines.push("</current_plan>");
  return lines.join("\n");
}

export function formatPlanReviewRequest(state: PlanState, options: { maxSteps?: number } = {}): string {
  const maxSteps = clampMaxSteps(options.maxSteps ?? 20);
  const visibleSteps = state.steps.slice(0, maxSteps);
  const lines = [
    "Plan ready for review.",
    "",
    `Objective: ${state.objective}`,
    "",
    "Steps:"
  ];
  if (visibleSteps.length === 0) {
    lines.push("No steps were created yet.");
  }
  visibleSteps.forEach((step, index) => {
    const current = step.id === state.currentStepId ? " current" : "";
    lines.push(`${index + 1}. [${step.status}${current}] ${step.id}: ${step.title}`);
    if (step.description) {
      lines.push(`   Description: ${step.description}`);
    }
    if (step.dependsOn && step.dependsOn.length > 0) {
      lines.push(`   Depends on: ${step.dependsOn.join(", ")}`);
    }
    if (step.completionCriteria && step.completionCriteria.length > 0) {
      lines.push(`   Completion: ${step.completionCriteria.join("; ")}`);
    }
  });
  if (state.steps.length > visibleSteps.length) {
    lines.push(`... ${state.steps.length - visibleSteps.length} more steps omitted.`);
  }
  lines.push("", "Please review this plan. Approve it to continue execution, or reply with changes.");
  return lines.join("\n");
}

function clampMaxSteps(value: number): number {
  if (!Number.isFinite(value)) {
    return 20;
  }
  return Math.min(100, Math.max(1, Math.floor(value)));
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}
