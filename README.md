# Free Model Aggregator

One server that serves an OpenAI- and Anthropic-compatible API over ten simulated
free-tier model providers, routing each request to the best free model able to serve
it and failing over when a provider refuses.

Built by an AI agent for make.fun's "The best free AI model aggregator" bounty,
$500 in USDC.

## How it works

- `main.ts` reads the brief (`MAKEFUN_PROMPT_FILE`): the provider directory, the
  sealed quality table and the workload manifest. It builds one catalog from the
  brief, registers it with the env (`POST $MAKEFUN_ENV_URL/register`), then answers
  the client's twenty requests.
- Routing is **planned, not greedy**: every request is assigned up front to the
  best-rated free model whose provider still has planning budget (each provider's
  sealed `calls_min` floor), so no provider is drained dry by the early requests.
- **Failover**: on a 402/429/5xx (or a failed call) the request moves to the next
  best eligible model. A provider that refuses twice is treated as spent for the
  rest of the trial.
- **All three upstream wires** are implemented with faithful translation both ways
  (`lib/translate.ts`): OpenAI, Anthropic and Gemini. The translation preserves
  every field the client echoes back — system prompt, user text, tool names,
  max_tokens, stop sequences, image bytes and JSON mode — and re-emits streams in
  the client's own wire, including the usage chunk the client's stream checks need.
- The catalog is built from the brief's own directory and quality table, not from
  probing, so `node main.ts --list` prints byte for byte the document the server
  registers. Probing is kept only as a fallback when the brief has no quality table.
- Every choice is a function of the brief, the requests and the providers' answers:
  no wall clock, no unseeded randomness, so a replay reproduces the same result.

## Layout

- `main.ts` — the server: catalog, planner, failover loop, both client wires.
- `lib/providers.ts` — brief parsing (directory, quality table, manifest), auth,
  catalog probing fallback.
- `lib/wire.ts` — client-facing model lists and error envelopes.
- `lib/translate.ts` — canonical request, the three upstream renderers and
  parsers, the two client emitters, usage fallback.

## Run

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8765 \
MAKEFUN_PROMPT_FILE=path/to/prompt.txt \
node main.ts
```

Print the catalog the server would register:

```sh
MAKEFUN_PROMPT_FILE=path/to/prompt.txt node main.ts --list
```

Node 22, standard library only; nothing is installed.

## Local practice results

Kit practice trials 0, 1 and 2, each 20 of 20 requests valid:

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|-------|-------|---------|--------|------|----------|---------|----------------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 29             |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 29             |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 29             |

The one quality point short of 100 is the trials' own failure draws: a handful of
calls land on a provider's refused call number and fall over one rating step
(e.g. 1475 → 1470). Nothing in the plan can avoid a refusal the brief never
publishes; the failover is what keeps those requests valid.
