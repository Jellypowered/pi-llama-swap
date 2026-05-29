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

| Key | Default | Description |
|-----|---------|-------------|
| `baseUrl` | `http://localhost:8080/v1` | API base URL (no trailing slash) |
| `providerId` | `llama-swap` | Provider ID for model registration |
| `defaultContextWindow` | `8192` | Fallback context window when `/v1/models` omits `n_ctx` |
| `propsTimeoutMs` | `120000` | Timeout for props discovery (ms) — increase for large models on slow networks. |
| `logPropsDiscovery` | `false` | Log verbose /props discovery progress (start, elapsed time, errors) to the notification area. |
| `apiKey` | `no-key` | API key forwarded to the provider |

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

This is useful for debugging slow or failing model metadata discovery.

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
