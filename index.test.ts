import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { streamSimple } from "@earendil-works/pi-ai/api/openai-completions";
import { normalizeContext } from "@earendil-works/pi-ai/utils/transcript";

import extension, {
	applyModelOverride,
	applyPropsMetadata,
	copyModelMetadata,
	fetchPropsMetadata,
	type ModelMetadataOverride,
} from "./index.ts";

const ninferOverride: ModelMetadataOverride = {
	contextWindow: 240_000,
	reasoning: true,
	thinkingLevelMap: {
		off: "none",
		minimal: null,
		low: "low",
		medium: "medium",
		high: null,
		xhigh: "xhigh",
		max: null,
	},
	compat: {
		thinkingFormat: "qwen",
		supportsReasoningEffort: true,
	},
	propsDiscovery: false,
};

function metadata() {
	return {
		reasoning: false,
		contextWindow: 8192,
		thinkingLevelMap: undefined,
		compat: undefined,
	};
}

async function waitFor(predicate: () => boolean): Promise<void> {
	for (let attempt = 0; attempt < 50; attempt++) {
		if (predicate()) {
			return;
		}
		await new Promise((resolve) => setTimeout(resolve, 2));
	}
	assert.fail("timed out waiting for asynchronous discovery");
}

test("reads valid llama.cpp /props metadata and preserves template thinking behavior", async () => {
	const result = await fetchPropsMetadata(
		"http://example.test/props",
		1000,
		async () =>
			new Response(
				JSON.stringify({
					default_generation_settings: { n_ctx: 131_072 },
					chat_template: "{% if enable_thinking %}",
				}),
				{ status: 200 },
			),
	);
	assert.equal(result.kind, "ok");
	if (result.kind !== "ok") {
		return;
	}

	const model = metadata();
	assert.equal(applyPropsMetadata(model, result.data), true);
	assert.equal(model.contextWindow, 131_072);
	assert.equal(model.reasoning, true);
	assert.equal(model.compat?.thinkingFormat, "qwen-chat-template");
	assert.equal(model.thinkingLevelMap?.low, null);
});

test("treats empty 404 and 405 /props responses as unsupported", async () => {
	for (const status of [404, 405] as const) {
		const result = await fetchPropsMetadata(
			"http://example.test/props",
			1000,
			async () => new Response(null, { status }),
		);
		assert.deepEqual(result, { kind: "unsupported", status });
	}
});

test("reports malformed, empty, and unexpected /props bodies without throwing", async () => {
	const bodies = ["{broken", "", "[]"];
	for (const body of bodies) {
		const result = await fetchPropsMetadata(
			"http://example.test/props",
			1000,
			async () => new Response(body, { status: 200 }),
		);
		assert.equal(result.kind, "invalid");
	}
});

test("preserves HTTP diagnostics and propagates network failures and timeouts", async () => {
	const httpResult = await fetchPropsMetadata(
		"http://example.test/props",
		1000,
		async () => new Response("backend unavailable", { status: 503 }),
	);
	assert.deepEqual(httpResult, {
		kind: "http-error",
		status: 503,
		detail: "backend unavailable",
	});

	await assert.rejects(
		fetchPropsMetadata("http://example.test/props", 1000, async () => {
			throw new TypeError("fetch failed");
		}),
		/fetch failed/,
	);

	await assert.rejects(
		fetchPropsMetadata("http://example.test/props", 5, async (_url, init) => {
			await new Promise((_resolve, reject) => {
				init?.signal?.addEventListener("abort", () =>
					reject(new DOMException("aborted", "AbortError")),
				);
			});
			throw new Error("unreachable");
		}),
		(error: Error) => error.name === "AbortError",
	);
});

test("per-model overrides win over defaults and discovered llama.cpp metadata", () => {
	const model = metadata();
	applyModelOverride(model, ninferOverride);
	assert.equal(model.contextWindow, 240_000);
	assert.equal(model.reasoning, true);
	assert.equal(model.compat?.thinkingFormat, "qwen");

	applyPropsMetadata(
		model,
		{
			default_generation_settings: { n_ctx: 262_144 },
			chat_template: "enable_thinking",
		},
		ninferOverride,
	);
	assert.equal(model.contextWindow, 240_000);
	assert.equal(model.compat?.thinkingFormat, "qwen");
});

test("metadata survives provider refresh snapshots and model switching", () => {
	const refreshed = metadata();
	applyModelOverride(refreshed, ninferOverride);
	const selected = metadata();
	copyModelMetadata(refreshed, selected);
	assert.deepEqual(selected, refreshed);

	const llama = metadata();
	applyPropsMetadata(llama, {
		default_generation_settings: { n_ctx: 32_768 },
		chat_template: "enable_thinking",
	});
	copyModelMetadata(llama, selected);
	assert.equal(selected.contextWindow, 32_768);
	assert.equal(selected.compat?.thinkingFormat, "qwen-chat-template");
	copyModelMetadata(refreshed, selected);
	assert.equal(selected.contextWindow, 240_000);
	assert.equal(selected.compat?.thinkingFormat, "qwen");
});

async function captureReasoningPayload(reasoning: "off" | "low" | "medium" | "xhigh") {
	let payload: Record<string, unknown> | undefined;
	const model = {
		id: "qwen3.8-27b-ninfer",
		name: "qwen3.8-27b-ninfer",
		api: "openai-completions" as const,
		provider: "llama-swap",
		baseUrl: "http://example.test/v1",
		reasoning: true,
		thinkingLevelMap: ninferOverride.thinkingLevelMap,
		input: ["text" as const],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 240_000,
		maxTokens: 4096,
		compat: ninferOverride.compat,
	};
	const context = normalizeContext({
		messages: [{ role: "user", content: "hello", timestamp: Date.now() }],
	});
	const stream = streamSimple(model, context, {
		apiKey: "no-key",
		...(reasoning === "off" ? {} : { reasoning }),
		fetch: async (_input, init) => {
			payload = JSON.parse(String(init?.body));
			return new Response(
				'data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"qwen3.8-27b","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
				{ headers: { "content-type": "text/event-stream" } },
			);
		},
	});
	for await (const _event of stream) {
		// Consume the stream so the request is issued.
	}
	assert.ok(payload);
	return payload;
}

test("NInfer reasoning serializes supported efforts and honors off", async () => {
	const off = await captureReasoningPayload("off");
	assert.equal(off.enable_thinking, false);
	assert.equal(off.reasoning_effort, undefined);

	for (const effort of ["low", "medium", "xhigh"] as const) {
		const payload = await captureReasoningPayload(effort);
		assert.equal(payload.enable_thinking, true);
		assert.equal(payload.reasoning_effort, effort);
		assert.equal(payload.chat_template_kwargs, undefined);
	}
});

test("mixed backends retain metadata across refresh and cache unsupported /props", async () => {
	const agentDir = await mkdtemp(join(tmpdir(), "pi-llama-swap-test-"));
	const originalAgentDir = process.env.PI_CODING_AGENT_DIR;
	const originalBaseUrl = process.env.LLAMA_BASE_URL;
	const originalFetch = globalThis.fetch;
	process.env.PI_CODING_AGENT_DIR = agentDir;
	process.env.LLAMA_BASE_URL = "http://swap.test/v1";
	const { propsDiscovery: _propsDiscovery, ...ninferDiscoveringOverride } = ninferOverride;
	await writeFile(
		join(agentDir, "pi-llama-swap.json"),
		JSON.stringify({ modelOverrides: { "ninfer-model": ninferDiscoveringOverride } }),
	);

	let ninferPropsCalls = 0;
	globalThis.fetch = async (input) => {
		const url = String(input);
		if (url.endsWith("/v1/models")) {
			return new Response(
				JSON.stringify({
					data: [{ id: "llama-model", meta: { n_ctx: 8192 } }, { id: "ninfer-model" }],
				}),
			);
		}
		if (url.endsWith("/upstream/ninfer-model/props")) {
			ninferPropsCalls++;
			return new Response(null, { status: 404 });
		}
		if (url.endsWith("/upstream/llama-model/props")) {
			return new Response(
				JSON.stringify({
					default_generation_settings: { n_ctx: 65_536 },
					chat_template: "enable_thinking",
				}),
			);
		}
		throw new Error(`unexpected URL: ${url}`);
	};

	const handlers = new Map<string, (...args: any[]) => any>();
	const registrations: any[] = [];
	const pi = {
		registerCommand() {},
		exec: async () => ({ code: 0, stdout: "", stderr: "" }),
		registerProvider: (_id: string, config: any) => registrations.push(structuredClone(config)),
		on: (name: string, handler: (...args: any[]) => any) => {
			handlers.set(name, handler);
		},
	};
	const ctx = { ui: { notify() {} } };

	try {
		await extension(pi as any);
		let models = registrations.at(-1).models;
		const ninferSelected = {
			...structuredClone(models.find((model: any) => model.id === "ninfer-model")),
			provider: "llama-swap",
		};
		await handlers.get("model_select")?.({ model: ninferSelected }, ctx);
		await waitFor(() => ninferPropsCalls === 1);
		await handlers.get("model_select")?.({ model: ninferSelected }, ctx);
		await new Promise((resolve) => setImmediate(resolve));
		assert.equal(ninferPropsCalls, 1, "an unsupported endpoint is queried only once per session");
		assert.equal(ninferSelected.contextWindow, 240_000);
		assert.equal(ninferSelected.compat.thinkingFormat, "qwen");

		const llamaSelected = {
			...structuredClone(models.find((model: any) => model.id === "llama-model")),
			provider: "llama-swap",
		};
		await handlers.get("model_select")?.({ model: llamaSelected }, ctx);
		await waitFor(() => llamaSelected.contextWindow === 65_536);
		assert.equal(llamaSelected.contextWindow, 65_536);
		assert.equal(llamaSelected.compat.thinkingFormat, "qwen-chat-template");

		await handlers.get("input")?.({ text: "/model" }, ctx);
		models = registrations.at(-1).models;
		assert.equal(models.find((model: any) => model.id === "ninfer-model").contextWindow, 240_000);
		assert.equal(models.find((model: any) => model.id === "llama-model").contextWindow, 65_536);
	} finally {
		globalThis.fetch = originalFetch;
		if (originalAgentDir === undefined) {
			delete process.env.PI_CODING_AGENT_DIR;
		} else {
			process.env.PI_CODING_AGENT_DIR = originalAgentDir;
		}
		if (originalBaseUrl === undefined) {
			delete process.env.LLAMA_BASE_URL;
		} else {
			process.env.LLAMA_BASE_URL = originalBaseUrl;
		}
		await rm(agentDir, { recursive: true, force: true });
	}
});
