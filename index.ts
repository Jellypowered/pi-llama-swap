/**
 * llama-swap provider for pi.
 *
 * Inspired by [huggingface/pi-llama](https://github.com/huggingface/pi-llama).
 *
 * Auto-discovers models from a running llama-swap and registers them
 * under the `llama-swap` provider.
 *
 * Config file (merged, project takes precedence):
 * - `~/.pi/agent/pi-llama-swap.json` (global)
 * - `./.pi/pi-llama-swap.json` (project-local)
 *
 * Example pi-llama-swap.json:
 * ```json
 * {
 *   "baseUrl": "http://localhost:8080/v1",
 *   "providerId": "llama-swap",
 *   "defaultContextWindow": 8192,
 *   "defaultMaxTokens": 8192,
 *   "propsTimeoutMs": 120000,
 *   "apiKey": "no-key"
 * }
 * ```
 *
 * Usage: `pi install github.com/Jellypowered/pi-llama-swap`
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";
import { Compile } from "typebox/compile";

// ---- Config ---------------------------------------------------------------

/** Shape of the pi-llama-swap.json configuration file. */
interface PiLlamaConfig {
	/** Base URL of the llama-swap API. */
	baseUrl?: string;
	/** Provider ID used to register models. */
	providerId?: string;
	/** Default context window when /v1/models and /props do not report it. */
	defaultContextWindow?: number;
	/** Default maximum output tokens when no per-model override is configured. */
	defaultMaxTokens?: number;
	/** Timeout in ms for props discovery requests. */
	propsTimeoutMs?: number;
	/** API key passed to the provider (default: "no-key"). */
	apiKey?: string;
	/** Log verbose /props discovery progress (start, elapsed time, errors). */
	logPropsDiscovery?: boolean;
	/** Per-model metadata and OpenAI compatibility overrides, keyed by model ID. */
	modelOverrides?: Record<string, ModelMetadataOverride>;
}

/** Metadata that cannot always be discovered through llama-swap. */
export interface ModelMetadataOverride {
	/** Exact logical context window exposed by this backend. */
	contextWindow?: number;
	/** Maximum output tokens Pi may request from this model. */
	maxTokens?: number;
	/** Whether the model supports controllable reasoning. */
	reasoning?: boolean;
	/** Map Pi thinking levels to backend-specific reasoning effort values. */
	thinkingLevelMap?: LlamaModel["thinkingLevelMap"];
	/** Pi OpenAI compatibility flags for this model/backend. */
	compat?: LlamaModel["compat"];
	/** Set false when this backend is known not to implement llama.cpp /props. */
	propsDiscovery?: boolean;
}

/** Sensible defaults for the global pi-llama-swap.json. */
const DEFAULT_GLOBAL_CONFIG: PiLlamaConfig = {
	baseUrl: "http://localhost:8080/v1",
	providerId: "llama-swap",
	defaultContextWindow: 8192,
	defaultMaxTokens: 8192,
	propsTimeoutMs: 120000,
	apiKey: "no-key",
	logPropsDiscovery: false,
};

/**
 * Create the global pi-llama-swap.json with defaults if it does not already exist.
 */
function ensureGlobalConfig(): void {
	const globalPath = join(getAgentDir(), "pi-llama-swap.json");
	if (existsSync(globalPath)) {
		return;
	}
	try {
		const agentDir = getAgentDir();
		if (!existsSync(agentDir)) {
			mkdirSync(agentDir, { recursive: true });
		}
		writeFileSync(globalPath, JSON.stringify(DEFAULT_GLOBAL_CONFIG, null, 2) + "\n", "utf-8");
		console.info(`[pi-llama-swap] Created global config: ${globalPath}`);
	} catch (err) {
		console.warn(`[pi-llama-swap] Failed to create global config: ${(err as Error).message}`);
	}
}

/**
 * Load merged config from global and project-local pi-llama-swap.json files.
 *
 * Project-local config (`./.pi/pi-llama-swap.json`) takes precedence over the
 * global config (`~/.pi/agent/pi-llama-swap.json`).
 */
function loadConfig(cwd: string): PiLlamaConfig {
	const globalPath = join(getAgentDir(), "pi-llama-swap.json");
	const projectPath = join(cwd, ".pi", "pi-llama-swap.json");

	let globalConfig: PiLlamaConfig = {};
	let projectConfig: PiLlamaConfig = {};

	// Load global config
	if (existsSync(globalPath)) {
		try {
			globalConfig = JSON.parse(readFileSync(globalPath, "utf-8"));
		} catch {
			console.warn(`[pi-llama-swap] Failed to parse global config: ${globalPath}`);
		}
	}

	// Load project-local config (overrides global)
	if (existsSync(projectPath)) {
		try {
			projectConfig = JSON.parse(readFileSync(projectPath, "utf-8"));
		} catch {
			console.warn(`[pi-llama-swap] Failed to parse project config: ${projectPath}`);
		}
	}

	// Merge: project overrides global
	return { ...globalConfig, ...projectConfig };
}

// ---- Defaults (overridden by config) --------------------------------------

// Fallback for /v1/models entries missing meta.n_ctx.
const DEFAULT_CONTEXT_WINDOW = 8192;
const DEFAULT_MAX_TOKENS = 8192;
const PROPS_TIMEOUT_MS = 120_000;

const ModelsResponseSchema = Type.Object({
	data: Type.Optional(
		Type.Array(
			Type.Object({
				id: Type.String(),
				status: Type.Optional(
					Type.Object({
						value: Type.Optional(
							Type.Union([
								Type.Literal("unloaded"),
								Type.Literal("loading"),
								Type.Literal("loaded"),
								Type.Literal("sleeping"),
								Type.Literal("unknown"),
							]),
						),
					}),
				),
				architecture: Type.Optional(
					Type.Object({
						input_modalities: Type.Optional(Type.Array(Type.String())),
					}),
				),
				meta: Type.Optional(
					Type.Object({
						n_ctx: Type.Optional(Type.Number()),
						n_params: Type.Optional(Type.Number()),
					}),
				),
			}),
		),
	),
});

const validateModelsResponse = Compile(ModelsResponseSchema);

const PropsResponseSchema = Type.Object({
	default_generation_settings: Type.Optional(
		Type.Object({
			n_ctx: Type.Optional(Type.Number()),
		}),
	),
	chat_template: Type.Optional(Type.String()),
});

const validatePropsResponse = Compile(PropsResponseSchema);

type ProviderModel = NonNullable<Parameters<ExtensionAPI["registerProvider"]>[1]["models"]>[number];
type LlamaModel = Extract<ProviderModel, { reasoning: boolean; contextWindow: number }>;
type ExtensionCtx = Parameters<Parameters<ExtensionAPI["on"]>[1]>[1];

// llama.cpp template thinking is boolean, so expose Pi's default off/medium toggle only.
const TEMPLATE_THINKING_LEVEL_MAP = {
	minimal: null,
	low: null,
	high: null,
	xhigh: null,
} satisfies NonNullable<LlamaModel["thinkingLevelMap"]>;

// Minimal shape needed to update both registered models and Pi's active model snapshot.
type MutableModelMetadata = {
	reasoning: boolean;
	contextWindow: number;
	maxTokens: number;
	thinkingLevelMap?: LlamaModel["thinkingLevelMap"];
	compat?: LlamaModel["compat"];
};

// Mark a model as using llama.cpp's chat_template_kwargs.enable_thinking control.
function applyTemplateThinkingSupport(model: MutableModelMetadata): void {
	model.reasoning = true;
	model.thinkingLevelMap = TEMPLATE_THINKING_LEVEL_MAP;
	model.compat = {
		...model.compat,
		// Despite the Pi enum name, this sends llama.cpp's generic
		// chat_template_kwargs.enable_thinking payload, not a Qwen-only option.
		thinkingFormat: "qwen-chat-template",
	};
}

/** Apply explicit configuration last so discovery can never overwrite it. */
export function applyModelOverride(
	model: MutableModelMetadata,
	override?: ModelMetadataOverride,
): void {
	if (!override) {
		return;
	}
	if (
		typeof override.contextWindow === "number" &&
		Number.isFinite(override.contextWindow) &&
		override.contextWindow > 0
	) {
		model.contextWindow = override.contextWindow;
	}
	if (
		typeof override.maxTokens === "number" &&
		Number.isFinite(override.maxTokens) &&
		override.maxTokens > 0
	) {
		model.maxTokens = override.maxTokens;
	}
	if (typeof override.reasoning === "boolean") {
		model.reasoning = override.reasoning;
	}
	if (override.thinkingLevelMap) {
		model.thinkingLevelMap = { ...override.thinkingLevelMap };
	}
	if (override.compat) {
		model.compat = { ...model.compat, ...override.compat };
	}
}

/** Keep Pi's selected-model snapshot in sync after provider re-registration. */
export function copyModelMetadata(
	source: MutableModelMetadata,
	target?: MutableModelMetadata,
): void {
	if (!target) {
		return;
	}
	target.contextWindow = source.contextWindow;
	target.maxTokens = source.maxTokens;
	target.reasoning = source.reasoning;
	target.thinkingLevelMap = source.thinkingLevelMap;
	target.compat = source.compat;
}

/** Merge llama.cpp metadata, then restore any explicit per-model settings. */
export function applyPropsMetadata(
	model: MutableModelMetadata,
	data: Static<typeof PropsResponseSchema>,
	override?: ModelMetadataOverride,
): boolean {
	let updated = false;
	const nCtx = data.default_generation_settings?.n_ctx;
	if (typeof nCtx === "number" && nCtx > 0) {
		model.contextWindow = nCtx;
		updated = true;
	}
	if (data.chat_template?.includes("enable_thinking") === true) {
		applyTemplateThinkingSupport(model);
		updated = true;
	}
	applyModelOverride(model, override);
	return updated;
}

export type PropsFetchResult =
	| { kind: "ok"; data: Static<typeof PropsResponseSchema> }
	| { kind: "unsupported"; status: 404 | 405 }
	| { kind: "http-error"; status: number; detail?: string }
	| { kind: "invalid"; detail: string };

/** Fetch optional llama.cpp metadata without attempting JSON parsing on HTTP errors. */
export async function fetchPropsMetadata(
	url: string,
	timeoutMs: number,
	fetchImpl: typeof fetch = fetch,
): Promise<PropsFetchResult> {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const response = await fetchImpl(url, { signal: controller.signal });
		if (response.status === 404 || response.status === 405) {
			return { kind: "unsupported", status: response.status };
		}

		const body = await response.text();
		if (!response.ok) {
			const detail = body.trim().slice(0, 300);
			return {
				kind: "http-error",
				status: response.status,
				...(detail ? { detail } : {}),
			};
		}
		if (!body.trim()) {
			return { kind: "invalid", detail: "empty response body" };
		}

		let data: unknown;
		try {
			data = JSON.parse(body);
		} catch (error) {
			return { kind: "invalid", detail: `malformed JSON: ${(error as Error).message}` };
		}
		if (!validatePropsResponse.Check(data)) {
			const detail = [...validatePropsResponse.Errors(data)]
				.map((error) => `${"path" in error ? error.path : ""} ${error.message}`)
				.join("; ");
			return { kind: "invalid", detail: detail || "unexpected JSON response" };
		}
		return { kind: "ok", data };
	} finally {
		clearTimeout(timer);
	}
}

export default async function (pi: ExtensionAPI) {
	let currentModels: LlamaModel[] = [];

	pi.registerCommand("llama-swap-version", {
		description: "Print llama-server --version output",
		handler: async (_args, ctx) => {
			const result = await pi.exec("llama-server", ["--version"]);
			const output = `${result.stderr ?? ""}\n${result.stdout ?? ""}`;
			const versionLine = output
				.split("\n")
				.map((l) => l.trim())
				.find((l) => /^version:\s/i.test(l));
			ctx.ui.notify(
				versionLine ?? `llama-server exited with code ${result.code}`,
				versionLine ? "info" : "error",
			);
		},
	});

	// Merge env vars (highest priority) with loaded config, falling back to defaults.
	ensureGlobalConfig();
	const config = loadConfig(process.cwd());

	const resolvedBaseUrl =
		process.env.LLAMA_BASE_URL ?? config.baseUrl ?? "http://localhost:8080/v1";
	const resolvedProviderId = config.providerId ?? "llama-swap";
	const resolvedApiKey = process.env.LLAMA_API_KEY ?? config.apiKey ?? "no-key";
	const resolvedContextWindow = config.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW;
	const resolvedMaxTokens = config.defaultMaxTokens ?? DEFAULT_MAX_TOKENS;
	const resolvedPropsTimeoutMs = config.propsTimeoutMs ?? PROPS_TIMEOUT_MS;

	const logPropsDiscovery = !!config.logPropsDiscovery;

	const baseUrl = resolvedBaseUrl.replace(/\/+$/, "");

	async function refreshProvider(): Promise<void> {
		try {
			const response = await fetch(`${baseUrl}/models`);
			if (!response.ok) {
				console.warn(`[llama-swap] ${baseUrl}/models returned ${response.status}`);
				return;
			}

			const payload: unknown = await response.json();
			if (!validateModelsResponse.Check(payload)) {
				const errors = [...validateModelsResponse.Errors(payload)]
					.map((e) => `${"path" in e ? e.path : ""} ${e.message}`)
					.join("; ");
				console.warn(`[llama-swap] invalid /models response: ${errors}`);
				return;
			}

			const previousById = new Map(currentModels.map((m) => [m.id, m]));

			currentModels = (payload.data ?? []).map((model) => {
				const previous = previousById.get(model.id);
				const isLoaded = model.status?.value === "loaded";
				const modalities = model.architecture?.input_modalities ?? ["text"];
				const input = modalities.filter(
					(m): m is "text" | "image" => m === "text" || m === "image",
				);
				const suffixes: string[] = [];
				if (input.includes("image")) {
					suffixes.push("(image)");
				}
				if (isLoaded) {
					suffixes.push("(loaded)");
				}
				const discovered = {
					id: model.id,
					name: suffixes.length > 0 ? `${model.id} ${suffixes.join(" ")}` : model.id,
					// Preserve metadata not present in /v1/models across provider refreshes.
					reasoning: previous?.reasoning ?? false,
					thinkingLevelMap: previous?.thinkingLevelMap,
					input,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: discoveredMetadata.has(model.id)
						? (previous?.contextWindow ?? model.meta?.n_ctx ?? resolvedContextWindow)
						: (model.meta?.n_ctx ?? previous?.contextWindow ?? resolvedContextWindow),
					maxTokens: previous?.maxTokens ?? resolvedMaxTokens,
					compat: previous?.compat,
				} as LlamaModel;
				applyModelOverride(discovered, config.modelOverrides?.[model.id]);
				return discovered;
			});

			if (currentModels.length === 0) {
				console.warn(`[llama-swap] no models returned from ${baseUrl}/models`);
				return;
			}

			pi.registerProvider(resolvedProviderId, {
				name: "llama-swap",
				baseUrl,
				apiKey: resolvedApiKey,
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			console.warn(`[llama-swap] failed to reach ${baseUrl}/models: ${(error as Error).message}`);
		}
	}

	const discoveredMetadata = new Set<string>();
	const unsupportedProps = new Set<string>();
	const pendingMetadata = new Set<string>();

	async function discoverModelMetadata(
		modelId: string,
		ctx?: ExtensionCtx,
		timeoutMs = resolvedPropsTimeoutMs,
		selectedModel?: MutableModelMetadata,
	): Promise<void> {
		const model = currentModels.find((m) => m.id === modelId);
		if (!model) {
			return;
		}
		const override = config.modelOverrides?.[modelId];
		applyModelOverride(model, override);
		if (override?.propsDiscovery === false) {
			unsupportedProps.add(modelId);
		}
		if (discoveredMetadata.has(modelId) || unsupportedProps.has(modelId)) {
			copyModelMetadata(model, selectedModel);
			return;
		}
		if (pendingMetadata.has(modelId)) {
			return;
		}

		pendingMetadata.add(modelId);
		// llama-swap exposes llama.cpp model props at /upstream/:id/props. Other
		// upstreams may not implement this optional endpoint.
		const propsBase = baseUrl.replace(/\/v1$/, "");
		const propsUrl = `${propsBase}/upstream/${encodeURIComponent(modelId)}/props`;
		const startedAt = performance.now();

		if (logPropsDiscovery) {
			ctx?.ui.notify(`[llama-swap] fetching /props for ${modelId}…`, "info");
		}

		try {
			const result = await fetchPropsMetadata(propsUrl, timeoutMs);
			const elapsed = Math.round(performance.now() - startedAt);

			if (result.kind === "unsupported") {
				unsupportedProps.add(modelId);
				copyModelMetadata(model, selectedModel);
				if (logPropsDiscovery) {
					ctx?.ui.notify(
						`[llama-swap] /props is unsupported for ${modelId} (HTTP ${result.status}, ${elapsed}ms)`,
						"info",
					);
				}
				return;
			}
			if (result.kind === "http-error") {
				const detail = result.detail ? `: ${result.detail}` : "";
				ctx?.ui.notify(
					`[llama-swap] /props for ${modelId} returned HTTP ${result.status}${detail} (${elapsed}ms to resolve /props)`,
					"error",
				);
				return;
			}
			if (result.kind === "invalid") {
				ctx?.ui.notify(
					`[llama-swap] invalid /props response for ${modelId}: ${result.detail} (${elapsed}ms to resolve /props)`,
					"error",
				);
				return;
			}

			const nCtx = result.data.default_generation_settings?.n_ctx;
			const updated = applyPropsMetadata(model, result.data, override);
			if (typeof nCtx === "number" && nCtx > 0 && logPropsDiscovery) {
				ctx?.ui.notify(
					`[llama-swap] contextWindow=${model.contextWindow} for ${modelId} (${elapsed}ms to resolve /props)`,
					"info",
				);
			}
			copyModelMetadata(model, selectedModel);
			discoveredMetadata.add(modelId);
			if (!updated) {
				if (logPropsDiscovery) {
					ctx?.ui.notify(
						`[llama-swap] no new metadata for ${modelId} (${elapsed}ms to resolve /props)`,
						"info",
					);
				}
				return;
			}
			pi.registerProvider(resolvedProviderId, {
				name: "llama-swap",
				baseUrl,
				apiKey: resolvedApiKey,
				api: "openai-completions",
				models: currentModels,
			});
		} catch (error) {
			const elapsed = Math.round(performance.now() - startedAt);
			const err = error as Error;
			const msg = err.name === "AbortError" ? "timeout" : err.message;
			ctx?.ui.notify(
				`[llama-swap] /props for ${modelId} failed: ${msg} (${elapsed}ms to resolve /props)`,
				"error",
			);
		} finally {
			pendingMetadata.delete(modelId);
		}
	}

	await refreshProvider();

	// Discover /props for models that are already active on session start.
	// This ensures context window and reasoning capabilities are discovered
	// even when resuming sessions or when a model is pre-selected.
	// Note: ctx.model may be undefined if the session started without a model selected,
	// or if model resolution happens after session_start fires.
	// In that case, the model_select event will handle it.
	const sessionStartHandler = (
		_event: { type: "session_start"; reason: string; previousSessionFile?: string },
		ctx: ExtensionCtx,
	) => {
		if (ctx.model?.provider === resolvedProviderId) {
			void discoverModelMetadata(ctx.model.id, ctx, resolvedPropsTimeoutMs, ctx.model);
		}
	};
	pi.on("session_start", sessionStartHandler);

	// If model_select fires (user selects a model), discover its props.
	pi.on("input", async (event) => {
		const trimmed = event.text.trim().toLowerCase();
		if (trimmed === "/model") {
			await refreshProvider();
		}
	});

	pi.on("model_select", (event, ctx) => {
		if (event.model.provider !== resolvedProviderId) {
			return;
		}
		void discoverModelMetadata(event.model.id, ctx, resolvedPropsTimeoutMs, event.model);
	});

	// Discover /props for already-active models because re-selecting them does not emit model_select.
	pi.on("before_provider_request", (event, ctx) => {
		const modelId = (event.payload as { model?: unknown })?.model;
		if (typeof modelId === "string") {
			const activeModel =
				ctx.model?.provider === resolvedProviderId && ctx.model.id === modelId
					? ctx.model
					: undefined;
			void discoverModelMetadata(modelId, ctx, resolvedPropsTimeoutMs, activeModel);
		}
	});
}
