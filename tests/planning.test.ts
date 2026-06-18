import assert from "node:assert/strict";
import test from "node:test";
import {
  ADD_PLAN_STEP_TOOL_NAME,
  APPROVE_PLAN_TOOL_NAME,
  CLEAR_PLAN_TOOL_NAME,
  CREATE_PLAN_TOOL_NAME,
  Planner,
  READ_PLAN_TOOL_NAME,
  SET_CURRENT_STEP_TOOL_NAME,
  UPDATE_PLAN_STEP_TOOL_NAME,
  buildPlanningInstructions,
  createPlanningTools,
  formatPlanSnapshot,
  hasOpenPlanSteps
} from "../src/planning/index.js";
import { ToolExecutor, ToolRegistry } from "../src/tools/registry.js";

test("planner creates, clones, updates, and validates step completion", () => {
  const planner = new Planner();
  const created = planner.create({
    objective: "Ship planner",
    steps: [
      { id: "inspect", title: "Inspect code" },
      { id: "implement", title: "Implement planner", dependsOn: ["inspect"], completionCriteria: ["Tests pass"] }
    ]
  });

  assert.equal(created.objective, "Ship planner");
  assert.equal(created.reviewStatus, "pending");
  assert.equal(created.revision, 1);
  assert.equal(created.currentStepId, "inspect");

  const approved = planner.approve({ note: "looks good" });
  assert.equal(approved.reviewStatus, "approved");
  assert.equal(typeof approved.reviewedAt, "string");

  const snapshot = planner.state;
  snapshot!.steps[0]!.title = "mutated outside";
  assert.equal(planner.state?.steps[0]?.title, "Inspect code");

  assert.throws(() => planner.setCurrentStep("implement"), /depends on incomplete step inspect/);
  assert.throws(() => planner.updateStep({ id: "inspect", status: "completed" }), /evidence/);

  const completed = planner.updateStep({
    id: "inspect",
    status: "completed",
    appendEvidence: ["Read src/agent/agent-loop.ts"]
  });
  assert.equal(completed.status, "completed");
  assert.equal(planner.state?.reviewStatus, "approved");

  const current = planner.setCurrentStep("implement");
  assert.equal(current.status, "in_progress");
  assert.equal(planner.state?.currentStepId, "implement");
  assert.equal(hasOpenPlanSteps(planner.state), true);
});

test("planner reports whether any plan steps are still open", () => {
  const planner = new Planner();
  planner.create({
    objective: "Close every step",
    steps: [
      { id: "done", title: "Done" },
      { id: "blocked", title: "Blocked" },
      { id: "cancelled", title: "Cancelled" }
    ]
  });

  assert.equal(hasOpenPlanSteps(undefined), false);
  assert.equal(hasOpenPlanSteps(planner.state), true);

  planner.updateStep({ id: "done", status: "completed", appendEvidence: ["verified"] });
  planner.updateStep({ id: "blocked", status: "blocked", appendEvidence: ["external blocker"] });
  planner.updateStep({ id: "cancelled", status: "cancelled" });

  assert.equal(hasOpenPlanSteps(planner.state), false);
});

test("planner rejects malformed initial state", () => {
  assert.throws(
    () =>
      new Planner({
        objective: "Bad",
        reviewStatus: "approved",
        revision: 1,
        createdAt: "2026-06-18T00:00:00.000Z",
        updatedAt: "2026-06-18T00:00:00.000Z",
        steps: [
          {
            id: "done",
            title: "Done",
            status: "completed",
            createdAt: "2026-06-18T00:00:00.000Z",
            updatedAt: "2026-06-18T00:00:00.000Z"
          }
        ]
      }),
    /evidence/
  );
});

test("planning tools share planner state and surface errors as tool results", async () => {
  const planner = new Planner();
  const executor = new ToolExecutor(new ToolRegistry(createPlanningTools(planner)));

  const createResult = await executor.execute({
    id: "call_create",
    name: CREATE_PLAN_TOOL_NAME,
    arguments: {
      objective: "Build planner",
      steps: [{ id: "one", title: "First step" }]
    }
  });
  assert.equal(createResult.isError, undefined);
  assert.match(createResult.content, /Plan created/);
  assert.equal(planner.state?.objective, "Build planner");
  assert.equal(planner.state?.reviewStatus, "pending");

  const readResult = await executor.execute({
    id: "call_read",
    name: READ_PLAN_TOOL_NAME,
    arguments: {}
  });
  assert.match(readResult.content, /First step/);

  const approveResult = await executor.execute({
    id: "call_approve",
    name: APPROVE_PLAN_TOOL_NAME,
    arguments: { note: "approved by user" }
  });
  assert.equal(approveResult.isError, undefined);
  assert.equal(planner.state?.reviewStatus, "approved");

  const badComplete = await executor.execute({
    id: "call_bad",
    name: UPDATE_PLAN_STEP_TOOL_NAME,
    arguments: { id: "one", status: "completed" }
  });
  assert.equal(badComplete.isError, true);
  assert.match(badComplete.content, /evidence/);

  const updateResult = await executor.execute({
    id: "call_update",
    name: UPDATE_PLAN_STEP_TOOL_NAME,
    arguments: { id: "one", status: "completed", appendEvidence: ["test evidence"] }
  });
  assert.equal(updateResult.isError, undefined);
  assert.equal(planner.state?.steps[0]?.status, "completed");

  const addResult = await executor.execute({
    id: "call_add",
    name: ADD_PLAN_STEP_TOOL_NAME,
    arguments: { id: "two", title: "Second step", setCurrent: true }
  });
  assert.equal(addResult.isError, undefined);
  assert.equal(planner.state?.currentStepId, "two");

  const setCurrent = await executor.execute({
    id: "call_current",
    name: SET_CURRENT_STEP_TOOL_NAME,
    arguments: { id: "two" }
  });
  assert.equal(setCurrent.isError, undefined);

  const clearResult = await executor.execute({
    id: "call_clear",
    name: CLEAR_PLAN_TOOL_NAME,
    arguments: {}
  });
  assert.equal(clearResult.isError, undefined);
  assert.equal(planner.state, undefined);
});

test("planning instructions and snapshot are static rules plus dynamic plan state", () => {
  const instructions = buildPlanningInstructions();
  assert.match(instructions.content, /create_plan/);
  assert.match(instructions.content, /read-only tools/);
  assert.doesNotMatch(instructions.content, /Ship planner/);

  const planner = new Planner();
  planner.create({
    objective: "Ship planner",
    steps: [{ id: "step_1", title: "Inspect <files>" }]
  });
  const snapshot = formatPlanSnapshot(planner.state);
  assert.match(snapshot ?? "", /<current_plan revision="1" review_status="pending">/);
  assert.match(snapshot ?? "", /review_status="pending"/);
  assert.match(snapshot ?? "", /Ship planner/);
  assert.match(snapshot ?? "", /Inspect &lt;files&gt;/);
});
