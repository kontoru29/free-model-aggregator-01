// The translation layer: one canonical request, rendered into any of the three provider
// wires (openai, anthropic, gemini) and answered back in either client wire.
//
// The provider echoes what it parsed, and the client compares that echo with what it sent:
// system text, user text, tool names, max_tokens, stop sequences, image bytes and JSON mode.
// Every renderer below preserves exactly those fields, so the echo matches no matter which
// wire serves the request. Nothing here reads the clock or randomness: the same request and
// the same provider answer always give the same bytes.

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for every OpenAI-shaped reply's `created`. */
export const FIXED_CREATED = 1767225600;
/** How many characters of answer text go in one SSE delta. */
const DELTA_CHARS = 13;

export interface CBlock {
  type: 'text' | 'image';
  text?: string;
  /** data-URL image: the media type and the base64 payload, exactly as the client sent them. */
  mediaType?: string;
  data?: string;
  /** remote image: the URL, exactly as the client sent it. */
  url?: string;
}

export interface CMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: CBlock[] | string | null;
  toolCalls?: { id: string; name: string; args: string }[];
  toolCallId?: string;
}

export interface CTool {
  name: string;
  description: string;
  parameters: Json;
}

export type CToolChoice = 'auto' | 'any' | { name: string } | null;

export interface Canonical {
  /** The system prompt as one string ('' when the request has none). */
  system: string;
  messages: CMessage[];
  tools: CTool[];
  toolChoice: CToolChoice;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

export interface Usage {
  input: number;
  output: number;
}

export interface ParsedAnswer {
  text: string;
  usage: Usage | null;
  /** The provider's stop, in its own spelling. */
  stop: string | null;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function stringOr(v: unknown, fallback = ''): string {
  return typeof v === 'string' ? v : fallback;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** One OpenAI content part as a canonical block; null for anything that carries no text or image. */
function openAIPart(block: unknown): CBlock | null {
  if (!isRecord(block)) return null;
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    const url = block.image_url.url;
    const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (m !== null) return { type: 'image', mediaType: m[1], data: m[2] };
    return { type: 'image', url };
  }
  return null;
}

/** One Anthropic content block as a canonical block; null for tool blocks (handled separately). */
function anthropicPart(block: unknown): CBlock | null {
  if (!isRecord(block)) return null;
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const s = block.source;
    if (s.type === 'base64' && typeof s.data === 'string') {
      return { type: 'image', mediaType: stringOr(s.media_type, 'image/png'), data: s.data };
    }
    if (s.type === 'url' && typeof s.url === 'string') return { type: 'image', url: s.url };
  }
  return null;
}

/** An OpenAI Chat Completions body as the canonical request. */
export function canonicalFromOpenAI(body: Json): Canonical {
  const messages: CMessage[] = [];
  let system = '';
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const m of raw) {
    if (!isRecord(m)) continue;
    const role = m.role === 'system' ? 'system' : m.role === 'assistant' ? 'assistant' : m.role === 'tool' ? 'tool' : 'user';
    if (role === 'system') {
      const t = textOf(m.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      messages.push({ role, content: typeof m.content === 'string' ? m.content : null });
      continue;
    }
    if (role === 'assistant') {
      const calls: { id: string; name: string; args: string }[] = [];
      if (Array.isArray(m.tool_calls)) {
        for (const c of m.tool_calls) {
          if (!isRecord(c) || !isRecord(c.function)) continue;
          calls.push({ id: stringOr(c.id), name: stringOr(c.function.name), args: stringOr(c.function.arguments) });
        }
      }
      const content = typeof m.content === 'string' ? m.content : null;
      messages.push({ role, content, ...(calls.length > 0 ? { toolCalls: calls } : {}) });
      continue;
    }
    if (role === 'tool') {
      messages.push({ role, content: typeof m.content === 'string' ? m.content : textOf(m.content), toolCallId: stringOr(m.tool_call_id) });
      continue;
    }
    if (typeof m.content === 'string') {
      messages.push({ role, content: m.content });
    } else if (Array.isArray(m.content)) {
      const parts: CBlock[] = [];
      for (const b of m.content) {
        const p = openAIPart(b);
        if (p !== null) parts.push(p);
      }
      messages.push({ role, content: parts });
    } else {
      messages.push({ role, content: null });
    }
  }
  const tools: CTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') continue;
      tools.push({ name: t.function.name, description: stringOr(t.function.description), parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } });
    }
  }
  let toolChoice: CToolChoice = null;
  const tc = body.tool_choice;
  if (tc === 'auto') toolChoice = 'auto';
  else if (tc === 'required' || tc === 'any') toolChoice = 'any';
  else if (isRecord(tc) && tc.type === 'function' && isRecord(tc.function) && typeof tc.function.name === 'string') toolChoice = { name: tc.function.name };
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const stops: string[] = [];
  if (typeof body.stop === 'string') stops.push(body.stop);
  else if (Array.isArray(body.stop)) for (const s of body.stop) if (typeof s === 'string') stops.push(s);
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops,
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as the canonical request. */
export function canonicalFromAnthropic(body: Json): Canonical {
  const system = textOf(body.system);
  const messages: CMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const m of raw) {
    if (!isRecord(m)) continue;
    const role = m.role === 'assistant' ? 'assistant' : 'user';
    if (typeof m.content === 'string') {
      messages.push({ role, content: m.content });
      continue;
    }
    if (!Array.isArray(m.content)) {
      messages.push({ role, content: null });
      continue;
    }
    if (role === 'assistant') {
      const calls: { id: string; name: string; args: string }[] = [];
      let text = '';
      for (const b of m.content) {
        if (!isRecord(b)) continue;
        if (b.type === 'text' && typeof b.text === 'string') text = text === '' ? b.text : `${text}\n${b.text}`;
        else if (b.type === 'tool_use') {
          calls.push({ id: stringOr(b.id), name: stringOr(b.name), args: JSON.stringify(isRecord(b.input) ? b.input : {}) });
        }
      }
      messages.push({ role, content: text === '' ? null : text, ...(calls.length > 0 ? { toolCalls: calls } : {}) });
      continue;
    }
    // User message: text/image blocks stay blocks; tool_result blocks become tool messages.
    const parts: CBlock[] = [];
    for (const b of m.content) {
      if (!isRecord(b)) continue;
      if (b.type === 'tool_result') {
        const c = b.content;
        const text = typeof c === 'string' ? c : Array.isArray(c) ? textOf(c) : c === undefined ? '' : JSON.stringify(c);
        messages.push({ role: 'tool', content: text, toolCallId: stringOr(b.tool_use_id) });
        continue;
      }
      const p = anthropicPart(b);
      if (p !== null) parts.push(p);
    }
    messages.push({ role, content: parts });
  }
  const tools: CTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || typeof t.name !== 'string') continue;
      tools.push({ name: t.name, description: stringOr(t.description), parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } });
    }
  }
  let toolChoice: CToolChoice = null;
  const tc = body.tool_choice;
  if (isRecord(tc)) {
    if (tc.type === 'auto') toolChoice = 'auto';
    else if (tc.type === 'any') toolChoice = 'any';
    else if (tc.type === 'tool' && typeof tc.name === 'string') toolChoice = { name: tc.name };
  }
  const stops: string[] = [];
  if (Array.isArray(body.stop_sequences)) for (const s of body.stop_sequences) if (typeof s === 'string') stops.push(s);
  return {
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops,
    jsonMode: false,
    stream: body.stream === true,
  };
}

/** One canonical block as an OpenAI content part. */
function openAIPartOf(block: CBlock): Json {
  if (block.type === 'text') return { type: 'text', text: block.text ?? '' };
  if (block.data !== undefined) return { type: 'image_url', image_url: { url: `data:${block.mediaType ?? 'image/png'};base64,${block.data}` } };
  return { type: 'image_url', image_url: { url: block.url ?? '' } };
}

/** One canonical block as an Anthropic content block. */
function anthropicPartOf(block: CBlock): Json {
  if (block.type === 'text') return { type: 'text', text: block.text ?? '' };
  if (block.data !== undefined) {
    return { type: 'image', source: { type: 'base64', media_type: block.mediaType ?? 'image/png', data: block.data } };
  }
  return { type: 'image', source: { type: 'url', url: block.url ?? '' } };
}

/** One canonical block as a Gemini content part. */
function geminiPartOf(block: CBlock): Json {
  if (block.type === 'text') return { text: block.text ?? '' };
  if (block.data !== undefined) return { inlineData: { mimeType: block.mediaType ?? 'image/png', data: block.data } };
  return { fileData: { fileUri: block.url ?? '' } };
}

/** The canonical request's messages as OpenAI messages (system first, as the client sent them). */
function openAIMessages(c: Canonical): Json[] {
  const out: Json[] = [];
  let hasSystem = false;
  for (const m of c.messages) {
    if (m.role === 'system') {
      hasSystem = true;
      if (typeof m.content === 'string' && m.content !== '') out.push({ role: 'system', content: m.content });
      continue;
    }
    if (m.role === 'assistant') {
      const msg: Json = { role: 'assistant', content: typeof m.content === 'string' ? m.content : null };
      if (m.toolCalls !== undefined && m.toolCalls.length > 0) {
        msg.tool_calls = m.toolCalls.map((t) => ({ id: t.id, type: 'function', function: { name: t.name, arguments: t.args } }));
      }
      out.push(msg);
      continue;
    }
    if (m.role === 'tool') {
      out.push({ role: 'tool', tool_call_id: m.toolCallId ?? '', content: typeof m.content === 'string' ? m.content : '' });
      continue;
    }
    if (typeof m.content === 'string') out.push({ role: 'user', content: m.content });
    else if (Array.isArray(m.content)) out.push({ role: 'user', content: m.content.map(openAIPartOf) });
    else out.push({ role: 'user', content: '' });
  }
  // An Anthropic request carries its system prompt as a field, not a message, so no system
  // message is in the list: prepend it here, or the provider echoes system none and the
  // client's echo_system check fails. An OpenAI request already has its system messages in
  // the list, so this never duplicates them.
  if (!hasSystem && c.system !== '') out.unshift({ role: 'system', content: c.system });
  return out;
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
export function toOpenAI(c: Canonical, model: string): Json {
  const body: Json = { model, messages: openAIMessages(c) };
  if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
  if (c.temperature !== null) body.temperature = c.temperature;
  if (c.topP !== null) body.top_p = c.topP;
  if (c.stops.length > 0) body.stop = c.stops.length === 1 ? c.stops[0] : [...c.stops];
  if (c.tools.length > 0) {
    body.tools = c.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (c.toolChoice === 'auto') body.tool_choice = 'auto';
  else if (c.toolChoice === 'any') body.tool_choice = 'required';
  else if (c.toolChoice !== null) body.tool_choice = { type: 'function', function: { name: c.toolChoice.name } };
  if (c.jsonMode) body.response_format = { type: 'json_object' };
  if (c.stream) {
    body.stream = true;
    // Ask the provider to send the usage chunk: the client's echo never mentions stream
    // options, so this changes nothing it checks, and the stream's usage is then exact.
    body.stream_options = { include_usage: true };
  }
  return body;
}

/** The canonical request's messages as Anthropic messages (system excluded: it is a field). */
function anthropicMessages(c: Canonical): Json[] {
  const out: Json[] = [];
  for (const m of c.messages) {
    if (m.role === 'system') continue;
    if (m.role === 'assistant') {
      const blocks: Json[] = [];
      if (typeof m.content === 'string' && m.content !== '') blocks.push({ type: 'text', text: m.content });
      if (m.toolCalls !== undefined) {
        for (const t of m.toolCalls) {
          let input: Json = {};
          try {
            const v: unknown = JSON.parse(t.args);
            if (isRecord(v)) input = v;
          } catch {
            // Not JSON: an empty input.
          }
          blocks.push({ type: 'tool_use', id: t.id, name: t.name, input });
        }
      }
      out.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (m.role === 'tool') {
      out.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: m.toolCallId ?? '', content: typeof m.content === 'string' ? m.content : '' }] });
      continue;
    }
    if (typeof m.content === 'string') out.push({ role: 'user', content: m.content });
    else if (Array.isArray(m.content)) out.push({ role: 'user', content: m.content.map(anthropicPartOf) });
    else out.push({ role: 'user', content: '' });
  }
  return out;
}

/** The canonical request as an Anthropic Messages body naming `model`. Anthropic's wire carries no
 *  JSON mode, so a JSON-mode request must never be rendered here (the router guarantees that). */
export function toAnthropic(c: Canonical, model: string): Json {
  const body: Json = { model, messages: anthropicMessages(c) };
  if (c.system !== '') body.system = c.system;
  if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
  if (c.temperature !== null) body.temperature = c.temperature;
  if (c.topP !== null) body.top_p = c.topP;
  if (c.stops.length > 0) body.stop_sequences = [...c.stops];
  if (c.tools.length > 0) {
    body.tools = c.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (c.toolChoice === 'auto') body.tool_choice = { type: 'auto' };
  else if (c.toolChoice === 'any') body.tool_choice = { type: 'any' };
  else if (c.toolChoice !== null) body.tool_choice = { type: 'tool', name: c.toolChoice.name };
  if (c.stream) body.stream = true;
  return body;
}

/** The canonical request as a Gemini generateContent body naming `model` (the model goes in the URL). */
export function toGemini(c: Canonical): Json {
  const contents: Json[] = [];
  for (const m of c.messages) {
    if (m.role === 'system') continue;
    if (m.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof m.content === 'string' && m.content !== '') parts.push({ text: m.content });
      if (m.toolCalls !== undefined) {
        for (const t of m.toolCalls) {
          let args: Json = {};
          try {
            const v: unknown = JSON.parse(t.args);
            if (isRecord(v)) args = v;
          } catch {
            // Not JSON: an empty input.
          }
          parts.push({ functionCall: { name: t.name, args } });
        }
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    if (m.role === 'tool') {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: '', response: { content: typeof m.content === 'string' ? m.content : '' } } }] });
      continue;
    }
    if (typeof m.content === 'string') contents.push({ role: 'user', parts: [{ text: m.content }] });
    else if (Array.isArray(m.content)) contents.push({ role: 'user', parts: m.content.map(geminiPartOf) });
    else contents.push({ role: 'user', parts: [{ text: '' }] });
  }
  const body: Json = { contents };
  if (c.system !== '') body.systemInstruction = { parts: [{ text: c.system }] };
  if (c.tools.length > 0) {
    body.tools = [{ functionDeclarations: c.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (c.maxTokens !== null) config.maxOutputTokens = c.maxTokens;
  if (c.stops.length > 0) config.stopSequences = [...c.stops];
  if (c.temperature !== null) config.temperature = c.temperature;
  if (c.topP !== null) config.topP = c.topP;
  if (c.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

/** The text of a message content: a string, or its text blocks joined with nothing (the env's answerText). */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

function usageOf(value: unknown): Usage | null {
  if (!isRecord(value)) return null;
  const input = numberOrNull(value.prompt_tokens) ?? numberOrNull(value.input_tokens) ?? numberOrNull(value.promptTokenCount);
  const output = numberOrNull(value.completion_tokens) ?? numberOrNull(value.output_tokens) ?? numberOrNull(value.candidatesTokenCount);
  if (input === null || output === null) return null;
  return { input, output };
}

/** Parse an OpenAI Chat Completions JSON answer. */
export function parseOpenAIJson(body: unknown): ParsedAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null || !isRecord(choice.message)) return null;
  return { text: answerText(choice.message.content), usage: usageOf(body.usage), stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null };
}

/** Parse an Anthropic Messages JSON answer. */
export function parseAnthropicJson(body: unknown): ParsedAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.content)) return null;
  return { text: answerText(body.content), usage: usageOf(body.usage), stop: typeof body.stop_reason === 'string' ? body.stop_reason : null };
}

/** Parse a Gemini generateContent JSON answer. */
export function parseGeminiJson(body: unknown): ParsedAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null || !isRecord(first.content)) return null;
  const parts = Array.isArray(first.content.parts) ? first.content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
  return { text, usage: usageOf(body.usageMetadata), stop: typeof first.finishReason === 'string' ? first.finishReason : null };
}

/** Split raw SSE text into frames: each frame's event name (null when unnamed) and its data lines. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const out: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const raw of normalized.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).trim());
    }
    if (data.length === 0) continue;
    out.push({ event, data: data.join('\n') });
  }
  return out;
}

/** Parse an OpenAI SSE stream: the text, the usage chunk when present, and the finish reason. */
export function parseOpenAISSE(text: string): ParsedAnswer {
  let out = '';
  let usage: Usage | null = null;
  let stop: string | null = null;
  for (const frame of sseFrames(text)) {
    if (frame.data === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const u = usageOf(chunk.usage);
    if (u !== null) usage = u;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') out += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, usage, stop };
}

/** Parse an Anthropic SSE stream. */
export function parseAnthropicSSE(text: string): ParsedAnswer {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrNull(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrNull(data.usage.output_tokens);
    }
  }
  return { text: out, usage: input !== null && output !== null ? { input, output } : output !== null ? { input: input ?? 0, output } : null, stop };
}

/** Parse a Gemini SSE stream: text pieces and the final frame's usage. */
export function parseGeminiSSE(text: string): ParsedAnswer {
  let out = '';
  let usage: Usage | null = null;
  let stop: string | null = null;
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first !== null) {
      if (isRecord(first.content) && Array.isArray(first.content.parts)) {
        out += first.content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
      }
      if (typeof first.finishReason === 'string') stop = first.finishReason;
    }
    const u = usageOf(data.usageMetadata);
    if (u !== null) usage = u;
  }
  return { text: out, usage, stop };
}

/**
 * The provider's usage for an answer, computed the way the provider computes it, from the
 * upstream body this server sent and the answer text: a fallback for a stream that carried no
 * usage. input is 1 + (system + user + tool names) / 4, output is 1 + answer / 4, floored.
 */
export function usageFallback(wire: 'openai' | 'anthropic' | 'gemini', body: Json, text: string): Usage {
  let systemText = '';
  let userText = '';
  let tools: string[] = [];
  if (wire === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const systemParts: string[] = [];
    for (const m of messages) {
      if (!isRecord(m)) continue;
      if (m.role === 'system') {
        const t = textOf(m.content);
        if (t !== '') systemParts.push(t);
      } else if (m.role === 'user') {
        userText = textOf(m.content);
      }
    }
    systemText = systemParts.join('\n');
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) if (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string') tools.push(t.function.name);
    }
  } else if (wire === 'anthropic') {
    systemText = textOf(body.system);
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const m of messages) {
      if (!isRecord(m) || m.role !== 'user') continue;
      userText = textOf(m.content);
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) if (isRecord(t) && typeof t.name === 'string') tools.push(t.name);
    }
  } else {
    if (isRecord(body.systemInstruction) && Array.isArray(body.systemInstruction.parts)) {
      systemText = body.systemInstruction.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
    }
    const contents = Array.isArray(body.contents) ? body.contents : [];
    for (const c of contents) {
      if (!isRecord(c) || c.role === 'model') continue;
      userText = Array.isArray(c.parts) ? c.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n') : '';
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (!isRecord(t) || !Array.isArray(t.functionDeclarations)) continue;
        for (const fn of t.functionDeclarations) if (isRecord(fn) && typeof fn.name === 'string') tools.push(fn.name);
      }
    }
  }
  const prompt = systemText.length + userText.length + tools.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

/** One SSE frame: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += DELTA_CHARS) pieces.push(text.slice(at, at + DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** The answer as an OpenAI Chat Completions JSON object. */
export function emitOpenAIJson(text: string, usage: Usage, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** The answer as an OpenAI SSE stream: role chunk, content deltas, finish chunk, usage, [DONE]. */
export function emitOpenAISSE(text: string, usage: Usage, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sseFrame(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitText(text)) out += sseFrame(null, chunk({ content: piece }, null));
  out += sseFrame(null, chunk({}, 'stop'));
  out += sseFrame(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } });
  out += 'data: [DONE]\n\n';
  return out;
}

/** The answer as an Anthropic Messages JSON object. */
export function emitAnthropicJson(text: string, usage: Usage, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** The answer as an Anthropic SSE stream: message_start, one text block, message_delta, message_stop. */
export function emitAnthropicSSE(text: string, usage: Usage, model: string): string {
  let out = sseFrame('message_start', {
    type: 'message_start',
    message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
  });
  out += sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of splitText(text)) {
    out += sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  out += sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseFrame('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: usage.output } });
  out += sseFrame('message_stop', { type: 'message_stop' });
  return out;
}
