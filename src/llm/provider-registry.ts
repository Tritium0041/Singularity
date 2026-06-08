import type { LlmProviderFactory } from "./types.js";

export class LlmProviderRegistry {
  private readonly providers = new Map<string, LlmProviderFactory>();

  register(provider: LlmProviderFactory): void {
    if (this.providers.has(provider.id)) {
      throw new Error(`LLM provider already registered: ${provider.id}`);
    }
    this.providers.set(provider.id, provider);
  }

  get(id: string): LlmProviderFactory {
    const provider = this.providers.get(id);
    if (!provider) {
      throw new Error(`No LLM provider registered for: ${id}`);
    }
    return provider;
  }

  list(): LlmProviderFactory[] {
    return [...this.providers.values()];
  }
}
