import { AnthropicMessagesClient, anthropicMessagesProvider } from "./anthropic-messages-client.js";
import { OpenAIChatCompletionsClient, openAIChatCompletionsProvider } from "./openai-chat-completions-client.js";
import { OpenAIResponsesClient, openAIResponsesProvider } from "./openai-responses-client.js";
import { LlmProviderRegistry } from "./provider-registry.js";
import type { LlmClient, LlmProviderFactoryOptions } from "./types.js";

export type BuiltInLlmProviderId = "openai-responses" | "openai-chat" | "anthropic";

export type EnvLlmFactoryOptions = LlmProviderFactoryOptions & {
  env?: NodeJS.ProcessEnv;
  provider?: string;
  model?: string;
  registry?: LlmProviderRegistry;
};

export type EnvLlmClient = {
  provider: BuiltInLlmProviderId;
  model: string;
  llm: LlmClient;
};

type ProviderEnvDefaults = {
  defaultModel: string;
  apiKeyEnv: string[];
  baseURLEnv: string[];
  modelEnv: string[];
};

const providerDefaults: Record<BuiltInLlmProviderId, ProviderEnvDefaults> = {
  "openai-responses": {
    defaultModel: "gpt-4.1-mini",
    apiKeyEnv: ["OPENAI_API_KEY"],
    baseURLEnv: ["OPENAI_BASE_URL"],
    modelEnv: ["OPENAI_MODEL"]
  },
  "openai-chat": {
    defaultModel: "gpt-4.1-mini",
    apiKeyEnv: ["OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY"],
    baseURLEnv: ["OPENAI_COMPAT_BASE_URL", "OPENAI_BASE_URL"],
    modelEnv: ["OPENAI_COMPAT_MODEL", "OPENAI_MODEL"]
  },
  anthropic: {
    defaultModel: "claude-3-5-sonnet-latest",
    apiKeyEnv: ["ANTHROPIC_API_KEY"],
    baseURLEnv: ["ANTHROPIC_BASE_URL"],
    modelEnv: ["ANTHROPIC_MODEL"]
  }
};

export function createDefaultLlmProviderRegistry(): LlmProviderRegistry {
  const registry = new LlmProviderRegistry();
  registry.register(openAIResponsesProvider);
  registry.register(openAIChatCompletionsProvider);
  registry.register(anthropicMessagesProvider);
  return registry;
}

export function createLlmClientFromEnv(options: EnvLlmFactoryOptions = {}): EnvLlmClient {
  const env = options.env ?? process.env;
  const provider = normalizeProvider(options.provider ?? env.LLM_PROVIDER ?? "openai-responses");
  const defaults = providerDefaults[provider];
  const model = firstNonEmpty(options.model, options.defaultModel, env.LLM_MODEL, firstEnv(env, defaults.modelEnv)) ?? defaults.defaultModel;
  const apiKey = firstNonEmpty(options.apiKey, env.LLM_API_KEY, firstEnv(env, defaults.apiKeyEnv));
  const baseURL = firstNonEmpty(options.baseURL, env.LLM_BASE_URL, firstEnv(env, defaults.baseURLEnv));
  const registry = options.registry ?? createDefaultLlmProviderRegistry();
  const factory = registry.get(provider);

  return {
    provider,
    model,
    llm: factory.create({
      apiKey,
      baseURL,
      defaultModel: model,
      defaultReasoning: options.defaultReasoning,
      fetchImpl: options.fetchImpl
    })
  };
}

export function createBuiltInLlmClient(provider: BuiltInLlmProviderId, options: LlmProviderFactoryOptions = {}): LlmClient {
  if (provider === "openai-responses") {
    return new OpenAIResponsesClient(options);
  }
  if (provider === "openai-chat") {
    return new OpenAIChatCompletionsClient(options);
  }
  return new AnthropicMessagesClient(options);
}

function normalizeProvider(value: string): BuiltInLlmProviderId {
  const normalized = value.trim().toLowerCase();
  if (normalized === "openai" || normalized === "responses" || normalized === "openai-responses") {
    return "openai-responses";
  }
  if (
    normalized === "chat" ||
    normalized === "openai-chat" ||
    normalized === "openai-compatible" ||
    normalized === "openai-completions" ||
    normalized === "chat-completions"
  ) {
    return "openai-chat";
  }
  if (normalized === "anthropic" || normalized === "anthropic-messages" || normalized === "claude") {
    return "anthropic";
  }
  throw new Error(`Unknown LLM_PROVIDER: ${value}. Expected openai-responses, openai-chat, or anthropic.`);
}

function firstEnv(env: NodeJS.ProcessEnv, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = env[key];
    if (value && value.trim()) {
      return value;
    }
  }
  return undefined;
}

function firstNonEmpty(...values: Array<string | undefined>): string | undefined {
  for (const value of values) {
    if (value && value.trim()) {
      return value;
    }
  }
  return undefined;
}
