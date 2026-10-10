# pi-llama-swap

llama-swap Pi extension — inspired by [huggingface/pi-llama](https://github.com/huggingface/pi-llama).

Auto-discovers models from a running llama-swap and registers them as the
`llama-swap` provider in pi.

## Install

**From the shell:**

```bash
pi install https://github.com/Jellypowered/pi-llama-swap
```

This clones to `~/.pi/agent/packages/pi-llama-swap/` and adds an entry to your pi
settings. Every future `pi` invocation auto-loads it.

**From inside an interactive pi session:**

```
!pi install https://github.com/Jellypowered/pi-llama-swap
```

Then run `/reload` (or restart pi) to load the extension.

**Dev mode:**

```bash
git clone https://github.com/Jellypowered/pi-llama-swap ~/code/pi-llama-swap
pi -e ~/code/pi-llama-swap/index.ts
```

`-e` loads the extension only for the current session, useful while
developing.

## Configuration

Create `~/.pi/agent/pi-llama-swap.json` for global config or
`./.pi/pi-llama-swap.json` for project-local overrides. Both are JSON files; the
project-local file takes precedence.

| Key                    | Default                    | Description                                                                                     |
| ---------------------- | -------------------------- | ----------------------------------------------------------------------------------------------- |
| `baseUrl`              | `http://localhost:8080/v1` | API base URL (no trailing slash)                                                                |
| `providerId`           | `llama-swap`               | Provider ID for model registration                                                              |
| `defaultContextWindow` | `8192`                     | Fallback context window when model metadata and `/props` omit it                                |
| `defaultMaxTokens`     | `8192`                     | Fallback maximum output tokens when no per-model override is configured                         |
| `propsTimeoutMs`       | `120000`                   | Timeout for props discovery (ms) — increase for large models on slow networks.                  |
| `logPropsDiscovery`    | `false`                    | Log verbose `/props` discovery progress (start, elapsed time, errors) to the notification area. |
| `modelOverrides`       | `{}`                       | Optional metadata/compatibility overrides keyed by exact model ID.                              |
| `apiKey`               | `no-key`                   | API key forwarded to the provider                                                               |

Each `modelOverrides.<model-id>` entry supports:

| Key                | Description                                                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `contextWindow`    | Exact logical context window for this model. Takes precedence over `/v1/models`, `/props`, and `defaultContextWindow`.                   |
| `maxTokens`        | Maximum output tokens Pi may request. Takes precedence over `defaultMaxTokens`.                                                          |
| `reasoning`        | Whether Pi should expose thinking controls for this model.                                                                               |
| `thinkingLevelMap` | Maps Pi levels (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) to backend effort strings. Use `null` for unsupported levels. |
| `compat`           | Pi OpenAI-completions compatibility settings, such as `thinkingFormat`, `supportsReasoningEffort`, or `maxTokensField`.                  |
| `propsDiscovery`   | Set to `false` for a backend known not to implement llama.cpp's `/props` endpoint.                                                       |

Overrides are optional. Models without an override retain the existing llama.cpp discovery
behavior. Exact model IDs are used deliberately; the extension does not guess a backend
from a model-name substring.

Environment variables take precedence over config file values:

- `LLAMA_BASE_URL` overrides `baseUrl`
- `LLAMA_API_KEY` overrides `apiKey`

#### Verbose discovery logging

Set `"logPropsDiscovery": true` to log verbose /props discovery progress to
the notification area. This is off by default to keep the notification area
clean. When enabled you'll see messages like:

```
[llama-swap] fetching /props for qwen3.6-35b-a3b-mtp…
[llama-swap] contextWindow=262144 for qwen3.6-35b-a3b-mtp (33ms to resolve /props)
```

This is useful for debugging slow or failing model metadata discovery. HTTP 404 and
405 mean that the selected upstream does not support this optional llama.cpp endpoint;
the extension remembers that result for the rest of the Pi session and continues using
`/v1/models` metadata plus any configured override.

### NInfer example

NInfer does not expose llama.cpp's `/props`, so configure metadata that cannot be
discovered. This example keeps the configured 240,000-token logical context and maps
only the effort levels accepted by NInfer:

```json
{
	"baseUrl": "http://localhost:8081/v1",
	"modelOverrides": {
		"qwen3.8-27b-ninfer": {
			"contextWindow": 240000,
			"maxTokens": 32768,
			"reasoning": true,
			"thinkingLevelMap": {
				"off": "none",
				"minimal": null,
				"low": "low",
				"medium": "medium",
				"high": null,
				"xhigh": "xhigh",
				"max": null
			},
			"compat": {
				"thinkingFormat": "qwen",
				"supportsReasoningEffort": true,
				"maxTokensField": "max_tokens"
			},
			"propsDiscovery": false
		}
	}
}
```

For Pi's OpenAI Completions adapter, `thinkingFormat: "qwen"` sends top-level
`enable_thinking`: `false` when thinking is off and `true` when it is enabled. With
`supportsReasoningEffort: true`, enabled requests also send the mapped top-level
`reasoning_effort` (`low`, `medium`, or `xhigh`). It does **not** send llama.cpp's
`chat_template_kwargs.enable_thinking` to NInfer. `maxTokens: 32768` makes Pi send
`max_tokens: 32768` (clamped down when the prompt needs more of the 240,000-token
context). Keep this aligned with NInfer's `--default-max-tokens` value. Thinking level
changes apply per request and do not require a server restart.

## Usage

```bash
# 1. Start llama-swap
llama-swap

# 2. Configure ~/.pi/agent/pi-llama-swap.json as needed

# 3. Launch pi
pi

# 4. Inside pi
/model              # pick models from your provider
```

### Environment Variable Override

You can override `baseUrl` and `apiKey` via environment variables without
editing the config file:

```bash
LLAMA_BASE_URL=http://10.0.0.5:3000/v1 LLAMA_API_KEY=secret pi
```
