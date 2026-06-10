import { DEFAULT_AGENT_INSTRUCTIONS } from "./default-instructions.js";
import type { PromptFragment, SystemPromptBackgroundOptions, SystemPromptBuilderOptions } from "./types.js";

export class PromptBuilder {
  buildConversationSystemPrompt(options: SystemPromptBuilderOptions): string | undefined {
    const fragments = [...(options.fragments ?? [])];
    const backgroundFragment = options.background === false ? undefined : this.buildBackgroundFragment(options.background ?? {});
    if (backgroundFragment) {
      fragments.unshift(backgroundFragment);
    }
    const defaultInstructions = options.defaultInstructions === false ? undefined : options.defaultInstructions ?? DEFAULT_AGENT_INSTRUCTIONS;
    return this.buildSystemPrompt(defaultInstructions, [toBasePromptFragment(options.basePrompt), ...fragments].filter(Boolean) as PromptFragment[]);
  }

  buildSystemPrompt(basePrompt: string | undefined, fragments: readonly PromptFragment[] = []): string | undefined {
    const parts = [basePrompt, ...fragments.map((fragment) => fragment.content)].filter((part): part is string => Boolean(part?.trim()));
    return parts.length > 0 ? parts.join("\n\n") : undefined;
  }

  buildBackgroundFragment(options: SystemPromptBackgroundOptions): PromptFragment | undefined {
    const sections = [
      renderEnvironmentContext(options),
      renderAvailableTools(options),
      ...renderExtraFragments(options.extra)
    ].filter((section): section is string => Boolean(section?.trim()));

    if (sections.length === 0) {
      return undefined;
    }

    return {
      id: "conversation-background",
      stable: true,
      content: sections.join("\n\n")
    };
  }
}

function toBasePromptFragment(basePrompt: string | undefined): PromptFragment | undefined {
  if (!basePrompt?.trim()) {
    return undefined;
  }
  return {
    id: "user-system-prompt",
    stable: true,
    content: basePrompt
  };
}

function renderEnvironmentContext(options: SystemPromptBackgroundOptions): string | undefined {
  const lines = ["<environment_context>"];
  if (options.includeCwd !== false && options.cwd) {
    lines.push(`  <cwd>${escapeXml(options.cwd)}</cwd>`);
  }
  if (options.includeCurrentDate !== false && options.currentDate) {
    lines.push(`  <current_date>${escapeXml(options.currentDate)}</current_date>`);
  }
  if (options.includeTimezone !== false && options.timezone) {
    lines.push(`  <timezone>${escapeXml(options.timezone)}</timezone>`);
  }
  if (options.includeShell !== false && options.shell) {
    lines.push(`  <shell>${escapeXml(options.shell)}</shell>`);
  }
  lines.push("</environment_context>");

  return lines.length > 2 ? lines.join("\n") : undefined;
}

function renderAvailableTools(options: SystemPromptBackgroundOptions): string | undefined {
  if (options.includeTools === false || !options.tools || options.tools.length === 0) {
    return undefined;
  }

  const lines = ["<available_tools>"];
  for (const tool of options.tools) {
    const name = escapeXml(tool.name);
    if (options.includeToolDescriptions && tool.description) {
      lines.push(`  <tool name="${name}">${escapeXml(tool.description)}</tool>`);
      continue;
    }
    lines.push(`  <tool name="${name}" />`);
  }
  lines.push("</available_tools>");
  return lines.join("\n");
}

function renderExtraFragments(extra: SystemPromptBackgroundOptions["extra"]): string[] {
  if (!extra) {
    return [];
  }
  if (typeof extra === "string") {
    return [extra];
  }
  return extra.map((fragment) => (typeof fragment === "string" ? fragment : fragment.content));
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
