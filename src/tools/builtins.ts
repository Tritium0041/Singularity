import type { AgentTool } from "./registry.js";

export const calculatorTool: AgentTool = {
  name: "calculator",
  description: "Evaluate a basic arithmetic expression.",
  access: "read",
  parameters: {
    type: "object",
    properties: {
      expression: {
        type: "string",
        description: "Arithmetic expression using numbers, parentheses, +, -, *, /, %, and **."
      }
    },
    required: ["expression"],
    additionalProperties: false
  },
  execute(args) {
    const expression = readStringProperty(args, "expression");
    if (!/^[\d\s+\-*/().%]+$/.test(expression)) {
      throw new Error("Calculator only accepts numbers, whitespace, parentheses, and arithmetic operators.");
    }

    const value = Function(`"use strict"; return (${expression});`)() as unknown;
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error("Expression did not produce a finite number.");
    }

    return { content: String(value) };
  }
};

export const mockWeatherTool: AgentTool = {
  name: "get_weather",
  description: "Return deterministic mock weather for a city.",
  access: "read",
  parameters: {
    type: "object",
    properties: {
      city: {
        type: "string",
        description: "City name."
      }
    },
    required: ["city"],
    additionalProperties: false
  },
  execute(args) {
    const city = readStringProperty(args, "city");
    return {
      content: `${city}: sunny, 24C, light breeze.`
    };
  }
};

function readStringProperty(args: unknown, key: string): string {
  if (!args || typeof args !== "object" || !(key in args)) {
    throw new Error(`Missing required argument: ${key}`);
  }

  const value = (args as Record<string, unknown>)[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`Argument ${key} must be a non-empty string.`);
  }

  return value;
}
