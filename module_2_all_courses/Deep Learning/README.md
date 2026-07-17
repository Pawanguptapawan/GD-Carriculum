# PILOT — Full Project Handbook
### Voice Copilot · Async Assistant · Grid Dynamics Capstone 2026

> **Portable Intelligent Listener for Open Tasking**
> A realtime voice pipeline where humans speak naturally, the agent listens continuously, distinguishes speakers, reacts immediately, delegates longer work asynchronously, and gates sensitive tools behind speaker authorization.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Problem We Are Solving](#2-what-problem-we-are-solving)
3. [System Architecture — Deep Dive](#3-system-architecture--deep-dive)
4. [Full Data Flow](#4-full-data-flow)
5. [Technology Stack (Final Choices)](#5-technology-stack-final-choices)
6. [Use Cases](#6-use-cases)
7. [Who Owns What — Per Person Breakdown](#7-who-owns-what--per-person-breakdown)
8. [Week-by-Week Plan](#8-week-by-week-plan)
9. [Edge Cases & How to Handle Them](#9-edge-cases--how-to-handle-them)
10. [Monolith Architecture Rationale](#10-monolith-architecture-rationale)
11. [Folder Structure](#11-folder-structure)
12. [Operational Rules](#12-operational-rules)
13. [Provider Swap Matrix](#13-provider-swap-matrix)

---

## 1. Project Overview

PILOT is a **realtime, always-on voice agent** built as an explicit pipeline — not a monolithic fused model. The core philosophy is to split the **interaction layer** (fast, real-time) from the **execution layer** (heavier, async), because no open-source model today fuses STT + VAD + tool-calling into one sub-200ms loop the way OpenAI's realtime APIs do.

| Attribute | Value |
|---|---|
| Team | 6 people — 3 DS, 2 FSE, 1 DevOps |
| Duration | 6 weeks |
| Budget | Open-source / free-tier only (no paid keys until post-capstone) |
| Language | Python (FastAPI + asyncio), vanilla JS frontend |
| Database | SQLite |
| Use cases | PPT Copilot, Customer Care / Flight Booking System |

---

## 2. What Problem We Are Solving

### Problems with existing systems

| Problem | Why it hurts |
|---|---|
| **Energy-threshold VAD** | Fires on silence, not meaning — cuts speakers mid-sentence during natural hesitations ("um", brief pauses) |
| **Monolithic pipelines** | One model tries to handle STT + reasoning + tools. No OSS model does this sub-200ms on a laptop |
| **No speaker awareness** | Systems treat all audio as one voice — can't distinguish who is speaking or restrict actions by role |
| **Unsafe tool execution** | Commands fire immediately — no speaker identity check, no confirmation for destructive actions |
| **No interruption model** | Agents can't cleanly cancel in-flight TTS or background jobs when a user speaks over them |
| **No async delegation** | Long-running work (filling a ticket, fetching a report) blocks the conversational loop |

### What PILOT does differently

```
"Humans speak naturally. PILOT listens continuously, knows who is speaking,
reacts immediately, delegates longer work asynchronously — safely."
```

- **Semantic VAD** fires on *linguistic completeness*, not silence
- **Dual-model split** — tiny fast front model + heavier async background agent
- **Speaker diarization + enrollment** — knows who said what, maps to roles
- **RBAC policy gate** — destructive actions require voice-confirmed authorization
- **Interrupt/queue concurrency** — barge-in kills TTS and cancels background jobs
- **Provider-swappable** — every subsystem behind an abstract interface

---

## 3. System Architecture — Deep Dive

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (SPA)                           │
│  WebRTC getUserMedia → PCM → WebSocket /ws/audio               │
│  WebSocket /ws/events ← live state push from backend           │
│  transcript_ui · tool_status · backlog_visualizer · enroll_ux  │
└────────────────────────┬────────────────────────────────────────┘
                         │ PCM audio (16kHz mono)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1 — AUDIO INGEST                      │
│                                                                 │
│  websocket_server.py (FastAPI)                                  │
│       │                                                         │
│       ├──► silero_vad.py     energy gate, 30ms frames          │
│       │         │                                               │
│       └──► smart_turn.py    Whisper-Tiny + linear classifier   │
│               │              semantic turn completion           │
│               ▼                                                 │
│           turn_q  (asyncio.Queue)                               │
│           TurnSegment(pcm, timestamp)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LAYER 2 — SPEAKER SEPARATION                   │
│                                                                 │
│  diarizer.py          Streaming Sortformer / pyannote fallback │
│       │               → pseudo-speaker labels: spk-0, spk-1    │
│       ▼                                                         │
│  identity_resolver.py cosine similarity vs enrollment store    │
│       │               → speaker_id, role, confidence           │
│       ▼                                                         │
│  enrollment_store.py  WeSpeaker embeddings in SQLite           │
│                                                                 │
│           labeled_turn_q  (asyncio.Queue)                       │
│           LabeledTurn(pcm, speaker_id, role, conf)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 3 — TRANSCRIPTION                      │
│                                                                 │
│  asr_worker.py        faster-whisper + distil-large-v3         │
│       │               batched transcription of labeled turns    │
│       ▼                                                         │
│  transcript_store.py  ring buffer (context) + SQLite log       │
│                                                                 │
│           transcript_q  (asyncio.Queue)                         │
│           TranscriptSpan(text, speaker_id, role, conf, ts)      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│             LAYER 4 — FRONT MODEL (ROUTER)                     │
│                                                                 │
│  front_model.py       Qwen3:8b via Ollama (local)              │
│       │               intent classification, barge-in policy    │
│       ▼                                                         │
│  intent_classifier.py RouteDecision(action, preamble, payload) │
│                                                                 │
│  Actions:                                                       │
│   • ignore         — conversational noise, filler              │
│   • respond_now    — say_preamble() → TTS                      │
│   • delegate       — ship to background agent                   │
└──────────┬──────────────────────────────────────┬──────────────┘
           │ respond_now                           │ delegate
           ▼                                       ▼
┌──────────────────────┐             ┌─────────────────────────────┐
│  LAYER 5a — TTS      │             │  LAYER 5b — BACKGROUND AGENT│
│                      │             │                             │
│  tts_engine.py       │             │  policy_gate.py             │
│  EdgeTTS (primary)   │             │  RBAC check + confirm loop  │
│  Kokoro (fallback)   │             │       │                     │
│       │              │             │       ▼                     │
│  WAV chunks → WS →   │             │  background_agent.py        │
│  browser playback    │             │  Gemini free tier / Groq    │
│       │              │             │  tool loop + retries        │
│  interrupt_handler   │             │       │                     │
│  cancel token        │             │  tool_registry.py           │
└──────────────────────┘             │  PPT / Tickets / KB / CRM   │
                                     └─────────────────────────────┘
                                                  │
                                     results stream → event_q → browser
```

---

## 4. Full Data Flow

### Step-by-step from microphone to tool execution

```
1. USER SPEAKS
   Browser mic (WebRTC getUserMedia)
   → 16kHz mono PCM chunks
   → WebSocket /ws/audio
   → raw_audio_q

2. VAD GATE
   silero_vad.py reads 30ms frames from raw_audio_q
   → filters silence, passes speech to smart_turn.py
   smart_turn.py runs Whisper-Tiny + linear classifier
   → scores: "is this turn semantically complete?"
   → when complete: emits TurnSegment(pcm, ts) onto turn_q

3. SPEAKER IDENTIFICATION
   diarizer.py consumes turn_q
   → Streaming Sortformer segments audio by speaker
   → emits pseudo-labels: spk-0, spk-1 etc.
   identity_resolver.py
   → WeSpeaker embedding of current segment
   → cosine similarity against enrollment_store
   → maps to (speaker_id, role, confidence) or spk-unknown
   → emits LabeledTurn onto labeled_turn_q

4. TRANSCRIPTION
   asr_worker.py consumes labeled_turn_q
   → faster-whisper + distil-large-v3
   → produces text for labeled audio span
   transcript_store.py dual-writes:
     - ring buffer in memory (context for front LLM)
     - append to SQLite sessions table
   → emits TranscriptSpan onto transcript_q

5. FRONT MODEL ROUTING
   front_model.py consumes transcript_q
   → Qwen3:8b via Ollama
   → receives: text, speaker_id, role, recent context
   → classifies intent:
       "ignore"      → drop, no action
       "respond_now" → call say_preamble()
       "delegate"    → build delegate(mode, payload)
   intent_classifier.py parses JSON output
   → RouteDecision(action, preamble_text, delegate_payload, mode)

6a. IMMEDIATE RESPONSE
    say_preamble() → tts_engine.py
    EdgeTTS (primary) or Kokoro (fallback)
    → streaming WAV chunks → WebSocket /ws/events → browser
    interrupt_handler.py holds cancel token
    → if user speaks while TTS playing:
        kill TTS stream, cancel tokens, drain queue

6b. BACKGROUND DELEGATION
    policy_gate.py checks RBAC:
    → is speaker_id allowed to call this tool?
    → is action destructive?
        if yes: read confirmation prompt aloud
        wait for affirmative from SAME speaker_id within timeout
        if different speaker confirms → fail closed
        if timeout → fail closed and log
    background_agent.py runs:
    → Gemini free tier API (primary) or Groq API (fallback)
    → tool loop with retries
    → tool_registry: PPT | Tickets | KB | CRM | Flight
    → streams state transitions to event_q → browser
    → on completion: results fed back to front model context

7. AUDIT
   Every tool call, every policy decision, every speaker hypothesis
   → audit_log.py → SQLite audit_log table
   (speaker_id, role, confidence, action, tool, decision, timestamp)
```

---

## 5. Technology Stack (Final Choices)

| Subsystem | Primary | Fallback | Notes |
|---|---|---|---|
| **Semantic VAD** | Smart Turn v3 (pipecat) | Silero VAD (energy gate only) | Smart Turn runs on Whisper-Tiny backbone, ~8M params, 10–100ms on CPU |
| **ASR** | faster-whisper + distil-large-v3 | — | Quantized; higher quality than base on handoffs |
| **Diarizer** | Streaming Sortformer (NVIDIA) | pyannote.audio | Sortformer handles 4 speakers; pyannote for >4 |
| **Speaker enrollment** | WeSpeaker embeddings | pyannote.audio embeddings | SQLite + numpy cosine similarity |
| **Front LLM** | Qwen3:8b (Ollama, local) | TBD | JSON-mode output; ~3–4B quantized target |
| **Background agent** | Gemini free tier API | Groq API key | Offloaded so laptops don't run two LLMs + ASR |
| **TTS** | EdgeTTS (Microsoft, free) | Kokoro (local) | EdgeTTS streams well; Kokoro for offline |
| **Backend** | FastAPI + asyncio (Python) | — | Monolith; uvicorn; single event loop |
| **Frontend** | Vanilla JS SPA | — | Served as FastAPI static files |
| **Audi...