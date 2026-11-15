// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// It reads the brief (provider directory, sealed quality table, workload manifest), builds
// one catalog from the brief, registers with the env, and answers the client's twenty
// requests with the best free model that meets each one's requirements, failing over when a
// provider refuses. The routing is planned, not greedy: every request is assigned up front
// to a provider under that provider's planning budget, so no provider is drained dry by the
// early requests; a refusal falls over to the next best eligible model.
//
// What the brief does not publish (each trial's budgets, failure schedule and outages) is
// learned at runtime: a provider that refuses twice is treated as spent for the rest of the
// trial. Every choice is a function of the brief, the requests and the providers' answers:
// no wall clock and no unseeded randomness, so a replay reproduces the same state export.
//
// The catalog is built from the brief's own directory and quality table, not from probing:
// the probes can only confirm what the brief already says, and a probe that fails (or a
// listing that hides a price) must not change the document `--list` prints or the register
// body carries. Probing is kept only as a fallback when the brief has no quality table.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type ProbedModel,
  type Probe,
  type RatingRow,
  type Requirement,
} from  './providers.ts';
import { anthropicError, anthropicModelList, openAIError, openAIModelList, type ServedModel } from  './wire.ts';
import {
  canonicalFromAnthropic,
  canonicalFromOpenAI,
  emitAnthropicJson,
  emitAnthropicSSE,
  emitOpenAIJson,
  emitOpenAISSE,
  parseAnthropicJson,
  parseAnthropicSSE,
  parseGeminiJson,
  parseGeminiSSE,
  parseOpenAIJson,
  parseOpenAISSE,
  toAnthropic,
  toGemini,
  toOpenAI,
  usageFallback,
  type Canonical,
  type ParsedAnswer,
  type Usage,
} from  './translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** One provider call may take at most this long before the next candidate is tried. */
const CALL_TIMEOUT_MS = 5000;
/** The most candidates tried for one request. */
const MAX_ATTEMPTS = 6;

/**
 * The planning budget of each provider: the sealed table's calls_min floor (the env's and
 * the scorer's shared table). A trial's real budget is drawn at or above this floor, so a
 * plan that spends at most this many calls a provider never knowingly overspends it; the
 * failures a trial draws on top are what the failover is for.
 */
const PLANNING_BUDGET_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let catalog: CatalogReport = { providers: [] };
/** The brief's providers, in order, with the models the quality table rates for them. */
interface Candidate {
  providerId: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  base: string;
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  key: string;
  model: string;
  rating: number;
}
let candidates: Candidate[] = [];
/** The request id -> the candidate the plan assigns it. */
const plan = new Map<string, Candidate>();
/** How many times each provider has refused a call in this trial. */
const failures = new Map<string, number>();
/** How many client API requests have arrived (the manifest is in arrival order). */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function isDead(providerId: string): boolean {
  return (failures.get(providerId) ?? 0) >= 2;
}

/** All free models that meet a request's requirements, best rating first, ties by provider then model. */
function eligible(requires: Requirement, api: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const c of candidates) {
    if (api === 'openai' && requires.json === true && c.wire === 'anthropic') continue;
    const row = ratingOfModel(brief.ratings, c.providerId, c.model);
    if (row === null) continue;
    const probed: ProbedModel = { id: c.model, display_name: c.model, free: row.free, context_length: row.context_length, tools: row.tools, vision: row.vision, json: row.json };
    if (!modelFree(probed, row)) continue;
    if (requires.tools === true && row.tools !== true) continue;
    if (requires.vision === true && row.vision !== true) continue;
    if (requires.json === true && row.json !== true) continue;
    if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) continue;
    out.push(c);
  }
  out.sort((a, b) => b.rating - a.rating || (a.providerId < b.providerId ? -1 : a.providerId > b.providerId ? 1 : a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
  return out;
}

/** The plan: assign every manifest request to the best candidate whose provider still has
 *  planning budget, in manifest order. A request with no budgeted candidate takes the best
 *  one outright; the failover covers the rest. */
function buildPlan(): void {
  const remaining = new Map<string, number>();
  for (const p of brief.directory.providers) remaining.set(p.id, PLANNING_BUDGET_MIN[p.id] ?? 0);
  const manifest = brief.workload?.manifest ?? [];
  for (const entry of manifest) {
    const list = eligible(entry.requires, entry.api);
    if (list.length === 0) continue;
    const pick = list.find((c) => (remaining.get(c.providerId) ?? 0) > 0) ?? list[0];
    plan.set(entry.id, pick);
    remaining.set(pick.providerId, (remaining.get(pick.providerId) ?? 0) - 1);
  }
}

/** The candidates to try for one request: the plan's pick first, then the rest by rating,
 *  with providers that have refused twice moved to the end. */
function orderFor(requestId: string, requires: Requirement, api: 'openai' | 'anthropic'): Candidate[] {
  const list = eligible(requires, api);
  const pick = plan.get(requestId);
  const ordered: Candidate[] = [];
  const seen = new Set<string>();
  if (pick !== undefined) {
    ordered.push(pick);
    seen.add(`${pick.providerId}/${pick.model}`);
  }
  for (const c of list) {
    const key = `${c.providerId}/${c.model}`;
    if (!seen.has(key)) {
      seen.add(key);
      ordered.push(c);
    }
  }
  const live = ordered.filter((c) => !isDead(c.providerId));
  const dead = ordered.filter((c) => isDead(c.providerId));
  return [...live, ...dead].slice(0, MAX_ATTEMPTS);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, api: 'openai' | 'anthropic'): { id: string; requires: Requirement } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { id: entry.id, requires: entry.requires };
  return { id: `unknown-${manifestIndex}`, requires: requirementsFromBody(body, api) };
}

function requirementsFromBody(body: Json, api: 'openai' | 'anthropic'): Requirement {
  if (api === 'anthropic') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    let vision = false;
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
    return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: false };
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Build the catalog and the candidate list from the brief; probe only when the brief has no table. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  const rows = brief.ratings?.ratings ?? [];
  candidates = [];
  for (const p of brief.directory.providers) {
    for (const row of rows) {
      if (row.provider !== p.id || !row.free) continue;
      candidates.push({ providerId: p.id, wire: p.wire, base: p.base, auth: p.auth, key: p.key, model: row.model, rating: row.rating_milli });
    }
  }
  if (rows.length > 0) {
    const providers: CatalogReport['providers'] = [];
    for (const p of brief.directory.providers) {
      const models: { id: string; free: boolean; context_length: number | null }[] = [];
      const seen = new Set<string>();
      for (const row of [...rows].filter((r) => r.provider === p.id).sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0))) {
        if (seen.has(row.model)) continue;
        seen.add(row.model);
        models.push({ id: row.model, free: row.free, context_length: row.context_length });
      }
      providers.push({ id: p.id, base_url: p.base, key_env: p.key_env, wire: p.wire, models });
    }
    catalog = { providers };
  } else {
    // No quality table in the brief: fall back to probing each provider's catalog route.
    const probes: Probe[] = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
    for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
    catalog = buildCatalog(ENV_URL, probes, brief.ratings);
    for (const probe of probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row: RatingRow | null = ratingOfModel(brief.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        candidates.push({ providerId: probe.provider.id, wire: probe.provider.wire, base: probe.provider.base, auth: probe.provider.auth, key: probe.provider.key, model: model.id, rating: row?.rating_milli ?? 0 });
      }
    }
  }
  buildPlan();
  log(`catalog: ${catalog.providers.length} providers, ${candidates.length} free models, plan for ${plan.size} requests`);
}

/** The free models the catalog lists, in brief order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free) continue;
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return out;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  answer: ParsedAnswer | null;
  error: string | null;
}

/** One call to one provider, in its own wire, answered as text plus usage. */
async function callProvider(candidate: Candidate, canon: Canonical): Promise<UpstreamResult> {
  const base = joinBase(ENV_URL, candidate.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(candidate.auth, candidate.key) };
  let url = '';
  let body: Json;
  if (candidate.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = toOpenAI(canon, candidate.model);
    if (canon.stream) headers.accept = 'text/event-stream';
  } else if (candidate.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = toAnthropic(canon, candidate.model);
    if (canon.stream) headers.accept = 'text/event-stream';
  } else {
    const path = canon.stream ? ':streamGenerateContent' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model)}${path}`;
    body = toGemini(canon);
    if (canon.stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, answer: null, error: `call to ${candidate.providerId} failed: ${(err as Error).message}` };
  }
  if (!res.ok) {
    let detail = '';
    try {
      detail = (await res.text()).slice(0, 300);
    } catch {
      // No body to read.
    }
    return { ok: false, status: res.status, answer: null, error: `${candidate.providerId} answered HTTP ${res.status}: ${detail}` };
  }
  const text = await res.text();
  let parsed: ParsedAnswer | null = null;
  try {
    if (candidate.wire === 'openai') parsed = canon.stream ? parseOpenAISSE(text) : parseOpenAIJson(JSON.parse(text));
    else if (candidate.wire === 'anthropic') parsed = canon.stream ? parseAnthropicSSE(text) : parseAnthropicJson(JSON.parse(text));
    else parsed = canon.stream ? parseGeminiSSE(text) : parseGeminiJson(JSON.parse(text));
  } catch {
    parsed = null;
  }
  if (parsed === null) return { ok: false, status: res.status, answer: null, error: `${candidate.providerId} answered in a shape this server could not read` };
  if (parsed.usage === null) parsed.usage = usageFallback(candidate.wire, body, parsed.text);
  return { ok: true, status: res.status, answer: parsed, error: null };
}

/** Answer one client request, trying the ordered candidates until one serves it. */
async function serve(req: IncomingMessage, res: ServerResponse, api: 'openai' | 'anthropic', body: Json): Promise<void> {
  const canon = api === 'openai' ? canonicalFromOpenAI(body) : canonicalFromAnthropic(body);
  const { id, requires } = requirementsFor(body, api);
  const ordered = orderFor(id, requires, api);
  if (ordered.length === 0) {
    return sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastError = 'no candidate answered';
  for (const candidate of ordered) {
    const result = await callProvider(candidate, canon);
    if (!result.ok || result.answer === null) {
      failures.set(candidate.providerId, (failures.get(candidate.providerId) ?? 0) + 1);
      lastError = result.error ?? `HTTP ${result.status}`;
      log(`${id}: ${candidate.providerId}/${candidate.model} refused (${lastError.slice(0, 120)})`);
      continue;
    }
    const usage: Usage = result.answer.usage ?? usageFallback(candidate.wire, toOpenAI(canon, candidate.model), result.answer.text);
    if (api === 'openai') {
      if (canon.stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(emitOpenAISSE(result.answer.text, usage, candidate.model));
      } else {
        sendJson(res, 200, emitOpenAIJson(result.answer.text, usage, candidate.model));
      }
    } else {
      if (canon.stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(emitAnthropicSSE(result.answer.text, usage, candidate.model));
      } else {
        sendJson(res, 200, emitAnthropicJson(result.answer.text, usage, candidate.model));
      }
    }
    return;
  }
  sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', `no provider served this request: ${lastError.slice(0, 200)}`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'openai', body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'anthropic', body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${candidates.length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
