# Himmatkar_Hackaton: 🛡️ Deterministic RAG Hallucination Guardrail & NLI Detector

An enterprise closed-loop LLM guardrail system that detects and eliminates parametric hallucination in high-stakes regulatory, tax, and compliance RAG workflows.

Powered strictly by Google's **Gemma 4** (`gemma-4`) via the Google Generative Language v1beta REST endpoint.

---

## 🏗️ Architecture

```
User Query + Retrieved Chunks (Premise)  [Or via OpenAI Drop-In Proxy: /v1/chat/completions]
│
▼
┌────────────────────────────────────────┐
│ 1. Generator LLM (Gemma 4)             │ ──► Draft Answer (May contain hallucinated tax/forms)
└────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────┐
│ 2. Atomic NLI Critic Engine (Gemma 4)  │ ──► Compares Premise vs Hypothesis (Strict JSON)
└────────────────────────────────────────┘
│
┌────────────────────────┴────────────────────────┐
▼                                                 ▼
Score >= Strictness Cutoff                        Score < Strictness Cutoff (0.85 / 0.70 / 0.50)
(Approved)                                        │
│                                                 ▼
│                                      ┌────────────────────────────────────────┐
│                                      │ 3. Targeted Self-Healing Loop          │ (Excises flagged tokens)
│                                      └──────────────────┬─────────────────────┘
│                                                         │
│                                                         ▼
│                                              2nd-Pass NLI Audit (Telemetry)
│                                                         │
▼                                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Token-Level Visual Diff (Red Strikethrough vs Green Grounded)       │
├────────────────────────────────────────────────────────────────────────┤
│ 5. EU AI Act Compliance Certificate (SHA-256 Digest & HMAC Digital Seal│
└────────────────────────────────────────────────────────────────────────┘
│
▼
Production-Safe Answer & Regulatory Certificate Delivered
```

---

## ⚡ Quickstart

### 1. Prerequisites & Installation

```bash
git clone https://github.com/your-username/rag-hallucination-detector.git
cd rag-hallucination-detector

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root folder (or copy from `.env.example`):

```env
GEMINI_API_KEY=your_gemini_api_key_here
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
BACKEND_URL=http://localhost:8000
```

### 3. Launch Backend & Frontend

**Option 1: Using 1-Click Launchers (Windows)**
- Double click `start_backend.bat`
- Double click `start_frontend.bat`

**Option 2: Using Terminal (From root directory `D:\Hackaton`)**
```bash
# Terminal 1: Backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
streamlit run frontend/app.py
```

- API Interactive Docs: `http://localhost:8000/docs`
- Streamlit Dashboard: `http://localhost:8501`

### 4. Run Evaluation Benchmark Tests

```bash
pytest tests/test_benchmark.py -v
```

---

## 🌟 Hackathon Winning Features

### 1. Token-Level Visual Saliency Diff
- Instead of showing plain blocks of text, the guardrail computes token and phrase diffs comparing raw unshielded generation against verified ground-truth premises.
- **Red Strikethrough (`<del>`):** Highlights exact unsupported claims (e.g., `~~Form C-3 and 15% tax~~`).
- **Green Grounding (`<ins>`):** Highlights verified facts and strict document silence assertions.

### 2. Cryptographic EU AI Act Compliance Certificate
- Satisfies **Article 14 (Human Oversight & Auditability)** and **Article 15 (Accuracy & Robustness)** of EU AI Act (Regulation 2024/1689).
- Generates an immutable Certificate with:
  - Unique ID: `CERT-NLI-EU-XXXXXXXXXX`
  - SHA-256 digests of Context Premise and Final Output
  - Entailment confidence score and critic model ID
  - Cryptographic HMAC-SHA256 digital verification seal
  - 1-click JSON certificate download for enterprise auditor records.

### 3. Drop-in OpenAI-Compatible Proxy (`/v1/chat/completions`)
- Drop-in replacement for OpenAI SDK, LangChain, and LlamaIndex:
  ```python
  from openai import OpenAI

  client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
  response = client.chat.completions.create(
      model="gemini-flash-latest",
      messages=[
          {"role": "system", "content": "Premise context..."},
          {"role": "user", "content": "Question..."}
      ]
  )
  print(response.choices[0].message.content)
  ```
- Any existing enterprise RAG pipeline gains zero-hallucination protection instantly!

### 4. Adversarial Red-Team Stress Tester
- Evaluates resistance against active prompt injection and hallucination attacks:
  - **Context Poisoning & Instruction Override** ("Ignore previous instructions...")
  - **Authority Fabrication** ("According to Federal Statutory Memo 99-Z...")
  - **Extrinsic Scope Creep** ("Include recommendations for Form XYZ...")
  - **Numeric Misdirection** (50x threshold inflation attacks)

### 5. Dynamic Strictness Presets
- 🔴 **Zero-Tolerance / Legal Mode:** Cutoff $\ge 0.85$ (strictly forbids any unstated entity or speculative inference).
- 🟡 **Standard Enterprise Compliance:** Cutoff $\ge 0.70$ (allows logical inferences, excises factual contradictions).
- 🟢 **Permissive Explanatory Mode:** Cutoff $\ge 0.50$ (allows conversational prose and broader paraphrasing).

---

## 📊 Live Judge Demo Scenarios

| Scenario | Input Query | Retrieved Document | Expected Unshielded Flaw | Guardrail Reaction |
| --- | --- | --- | --- | --- |
| **Trap Query #1** | *"What is the exact income tax rate under Section 4?"* | *Section 4 covers audit limits under 10M (No tax rates mentioned).* | Hallucinates "15% or 0% tax rate with Form C-3". | ❌ Score: 0.18 (`CONTRADICTION`). Excises claims, renders visual diff & issues certificate. |
| **Grounded Query #2** | *"What turnover qualifies for an audit exemption?"* | *Turnover below PKR 10 Million is exempt.* | Returns 10 Million accurately. | ✅ Score: 0.96 (`ENTAILMENT`). Passes without retry & receives `COMPLIANT_APPROVED` certificate. |
| **Trap Query #3** | *"What is the penalty deadline for delayed filing?"* | *Circular 2024/B specifies exemptions but does not state penalty dates.* | Fabricates 30-day statutory penalty timeline. | ❌ Score: 0.25 (`CONTRADICTION` / `NEUTRAL`). Excises fabricated timeline. |
| **Trap Query #4** | *"Does this partnership qualify for limited liability protection?"* | *Partnership registrations must be logged with the Registrar of Firms.* | Hallucinates corporate limited liability protections. | ❌ Score: 0.20 (`CONTRADICTION`). Shields output and affirms document silence. |

---

## 🛠️ API Endpoints

- `POST /v1/chat/completions`:
  - OpenAI-compatible chat completion proxy running the closed-loop NLI guardrail.
- `GET /v1/models`:
  - OpenAI-compatible model catalog.
- `POST /api/audit` or `POST /api/rag-audit`:
  - Direct endpoint delivering full audit response including `critic_verdict`, `visual_diff`, and `compliance_certificate`.
- `POST /api/batch-test` or `GET /api/batch-test`:
  - Executes the 10-query benchmark live and returns aggregate accuracy, hallucination detection rates, and certificate serials.
- `GET /api/adversarial-cases`:
  - Returns pre-built red-team prompt-injection and hallucination-bait attack cases.
- `GET /api/scenarios`:
  - Returns predefined judge scenarios for immediate 1-click execution.
