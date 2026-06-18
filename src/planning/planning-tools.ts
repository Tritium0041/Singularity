import type { AgentTool } from "../tools/registry.js";
import type { Planner } from "./planner.js";
import { PLAN_STEP_STATUSES, type PlanStepStatus } from "./types.js";

export const CREATE_PLAN_TOOL_NAME = "create_plan";
export const READ_PLAN_TOOL_NAME = "read_plan";
export const UPDATE_PLAN_STEP_TOOL_NAME = "update_plan_step";
export const ADD_PLAN_STEP_TOOL_NAME = "add_plan_step";
export const SET_CURRENT_STEP_TOOL_NAME = "set_current_step";
export const APPROVE_PLAN_TOOL_NAME = "approve_plan";
export const CLEAR_PLAN_TOOL_NAME = "clear_plan";

export function createPlanningTools(planner: Planner): AgentTool[] {
  return [
    {
      name: CREATE_PLAN_TOOL_NAME,
      description: "Create or replace the structured plan for the current user goal.",
      access: "planner",
      executionMode: "sequential",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          objective: { type: "string", description: "Current user goal or task objective." },
          steps: {
            type: "array",
            description: "Ordered plan steps.",
            items: {
              type: "object",
              additionalProperties: false,
              properties: {
                id: { type: "string" },
                title: { type: "string" },
                description: { type: "string" },
                dependsOn: { type: "array", items: { type: "string" } },
                completionCriteria: { type: "array", items: { type: "string" } },
                evidence: { type: "array", items: { type: "string" } }
              },
              required: ["title"]
            }
          },
          currentStepId: { type: "string", description: "Optional current step id." }
        },
        required: ["objective"]
      },
      execute(args) {
        const input = readRecord(args);
        const state = planner.state ? planner.replace(readCreatePlanArgs(input)) : planner.create(readCreatePlanArgs(input));
        return planToolResult("Plan created.", state);
      }
    },
    {
      name: READ_PLAN_TOOL_NAME,
      description: "Read the current plan and step status.",
      access: "planner",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {}
      },
      execute() {
        const state = planner.state;
        return planToolResult(state ? "Current plan." : "No active plan.", state ?? { plan: undefined });
      }
    },
    {
      name: UPDATE_PLAN_STEP_TOOL_NAME,
      description: "Update a plan step status, fields, completion criteria, evidence, or current-step marker.",
      access: "planner",
      executionMode: "sequential",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          id: { type: "string" },
          title: { type: "string" },
          description: { type: "string" },
          status: { type: "string", enum: [...PLAN_STEP_STATUSES] },
          dependsOn: { type: "array", items: { type: "string" } },
          completionCriteria: { type: "array", items: { type: "string" } },
          evidence: { type: "array", items: { type: "string" } },
          appendEvidence: { type: "array", items: { type: "string" } },
          setCurrent: { type: "boolean" }
        },
        required: ["id"]
      },
      execute(args) {
        const step = planner.updateStep(readUpdateStepArgs(args));
        return planToolResult(`Updated plan step ${step.id}.`, { step, plan: planner.state });
      }
    },
    {
      name: ADD_PLAN_STEP_TOOL_NAME,
      description: "Add a new step to the current plan.",
      access: "planner",
      executionMode: "sequential",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          id: { type: "string" },
          title: { type: "string" },
          description: { type: "string" },
          status: { type: "string", enum: [...PLAN_STEP_STATUSES] },
          dependsOn: { type: "array", items: { type: "string" } },
          completionCriteria: { type: "array", items: { type: "string" } },
          evidence: { type: "array", items: { type: "string" } },
          setCurrent: { type: "boolean" }
        },
        required: ["title"]
      },
      execute(args) {
        const input = readRecord(args);
        const step = planner.addStep({
          id: readOptionalString(input, "id"),
          title: readRequiredString(input, "title"),
          description: readOptionalString(input, "description"),
          status: readOptionalStatus(input, "status"),
          dependsOn: readOptionalStringArray(input, "dependsOn"),
          completionCriteria: readOptionalStringArray(input, "completionCriteria"),
          evidence: readOptionalStringArray(input, "evidence"),
          setCurrent: readOptionalBoolean(input, "setCurrent")
        });
        return planToolResult(`Added plan step ${step.id}.`, { step, plan: planner.state });
      }
    },
    {
      name: SET_CURRENT_STEP_TOOL_NAME,
      description: "Set the current step. Pending steps become in_progress when dependencies are complete.",
      access: "planner",
      executionMode: "sequential",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          id: { type: "string" }
        },
        required: ["id"]
      },
      execute(args) {
        const step = planner.setCurrentStep(readRequiredString(readRecord(args), "id"));
        return planToolResult(`Current plan step is ${step.id}.`, { step, plan: planner.state });
      }
    },
    {
      name: APPROVE_PLAN_TOOL_NAME,
      description: "Approve the current plan after the user has reviewed it, allowing execution tools to become available.",
      access: "planner",
      executionMode: "sequential",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          note: { type: "string", description: "Optional user review note or approval text." }
        }
      },
      execute(args) {
        const input = readRecord(args);
        const state = planner.approve({
          note: readOptionalString(input, "note")
        });
        return planToolResult("Plan approved.", state);
      }
    },
    {
      name: CLEAR_PLAN_TOOL_NAME,
      description: "Clear the current plan when the user goal changes or the task is complete.",
      access: "planner",
      executionMode: "sequential",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {}
      },
      execute() {
        planner.clear();
        return planToolResult("Plan cleared.", { plan: undefined });
      }
    }
  ];
}

function readCreatePlanArgs(input: Record<string, unknown>) {
  return {
    objective: readRequiredString(input, "objective"),
    steps: readPlanSteps(input.steps),
    currentStepId: readOptionalString(input, "currentStepId")
  };
}

function readPlanSteps(value: unknown) {
  if (value === undefined) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new Error("Argument steps must be an array.");
  }
  return value.map((item, index) => {
    const step = readRecord(item, `steps[${index}]`);
    return {
      id: readOptionalString(step, "id"),
      title: readRequiredString(step, "title"),
      description: readOptionalString(step, "description"),
      dependsOn: readOptionalStringArray(step, "dependsOn"),
      completionCriteria: readOptionalStringArray(step, "completionCriteria"),
      evidence: readOptionalStringArray(step, "evidence")
    };
  });
}

function readUpdateStepArgs(args: unknown) {
  const input = readRecord(args);
  return {
    id: readRequiredString(input, "id"),
    title: readOptionalString(input, "title"),
    description: readOptionalString(input, "description"),
    status: readOptionalStatus(input, "status"),
    dependsOn: readOptionalStringArray(input, "dependsOn"),
    completionCriteria: readOptionalStringArray(input, "completionCriteria"),
    evidence: readOptionalStringArray(input, "evidence"),
    appendEvidence: readOptionalStringArray(input, "appendEvidence"),
    setCurrent: readOptionalBoolean(input, "setCurrent")
  };
}

function planToolResult(message: string, details: unknown) {
  return {
    content: `${message}\n${JSON.stringify(details, null, 2)}`,
    details
  };
}

function readRecord(value: unknown, label = "Tool arguments"): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function readRequiredString(input: Record<string, unknown>, key: string): string {
  const value = input[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`Argument ${key} must be a non-empty string.`);
  }
  return value.trim();
}

function readOptionalString(input: Record<string, unknown>, key: string): string | undefined {
  const value = input[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new Error(`Argument ${key} must be a string.`);
  }
  return value;
}

function readOptionalBoolean(input: Record<string, unknown>, key: string): boolean | undefined {
  const value = input[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new Error(`Argument ${key} must be a boolean.`);
  }
  return value;
}

function readOptionalStringArray(input: Record<string, unknown>, key: string): string[] | undefined {
  const value = input[key];
  if (value === undefined) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new Error(`Argument ${key} must be an array of strings.`);
  }
  return value;
}

function readOptionalStatus(input: Record<string, unknown>, key: string): PlanStepStatus | undefined {
  const value = input[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string" || !(PLAN_STEP_STATUSES as readonly string[]).includes(value)) {
    throw new Error(`Argument ${key} must be one of: ${PLAN_STEP_STATUSES.join(", ")}.`);
  }
  return value as PlanStepStatus;
}
