import type { PromptFragment } from "../context/types.js";
import type { Skill, SkillLoadResult } from "./types.js";

export const SKILLS_PROMPT_FRAGMENT_ID = "available-skills";

export function buildSkillsPromptFragment(result: SkillLoadResult): PromptFragment | undefined {
  const content = renderAvailableSkills(result.skills);
  if (!content) {
    return undefined;
  }
  return {
    id: SKILLS_PROMPT_FRAGMENT_ID,
    stable: true,
    content
  };
}

export function renderAvailableSkills(skills: readonly Skill[]): string | undefined {
  const visibleSkills = skills.filter((skill) => !skill.disableModelInvocation);
  if (visibleSkills.length === 0) {
    return undefined;
  }

  const lines = [
    "<available_skills>",
    "  <instructions>When a task matches a skill description, read the skill file first, then follow its instructions. Resolve relative files from the skill file directory.</instructions>"
  ];
  for (const skill of visibleSkills) {
    lines.push(
      `  <skill name="${escapeXml(skill.name)}" location="${escapeXml(skill.filePath)}">${escapeXml(skill.description)}</skill>`
    );
  }
  lines.push("</available_skills>");
  return lines.join("\n");
}

export function formatSkillInvocation(skill: Skill, body: string, args?: string): string {
  const lines = [
    `<skill name="${escapeXml(skill.name)}" location="${escapeXml(skill.filePath)}">`,
    body.trimEnd(),
    "</skill>"
  ];
  if (args?.trim()) {
    lines.push("", `<skill_arguments>${escapeXml(args.trim())}</skill_arguments>`);
  }
  return lines.join("\n");
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
