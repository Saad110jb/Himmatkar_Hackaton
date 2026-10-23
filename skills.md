# Agent Skills & Subsystem Capabilities: NLI Hallucination Guardrail (Powered by Gemma 4)

## 1. Skill: Context-Bound Generation (The Generator: Gemma 4)
- **Role:** Generates regulatory and legal responses strictly from provided chunk premises using Google Gemma 4.
- **Protocol:**
  - Ingests `context` (verified text) and `question`.
  - Enforces negative constraints: *"If the detail cannot be verified from the premise, state 'Not available in provided documents'"*.
  - Operates at `temperature=0.1` to reduce creative variance.

## 2. Skill: Atomic Natural Language Inference (The NLI Critic: Gemma 4)
- **Role:** Impartial, zero-tolerance fact-checker comparing generated outputs against the ground-truth premise via Gemma 4.
- **Protocol:**
  - Maps `context` -> `PREMISE` and `generator_output` -> `HYPOTHESIS`.
  - Determines logical entailment:
    - **ENTAILMENT (0.80 - 1.0):** Every claim is supported directly by the text.
    - **NEUTRAL (0.30 - 0.70):** Introduces external real-world facts, missing forms, or unmentioned dates (extrinsic hallucination).
    - **CONTRADICTION (0.0 - 0.29):** Directly conflicts with premise figures or stipulations.
  - Returns structured JSON validated via Pydantic:
    ```json
    {
      "entailment_score": 0.15,
      "verdict": "CONTRADICTION",
      "hallucinated_part": "Form C-3 and 0% tax rate",
      "confidence": 0.95
    }
    ```

## 3. Skill: Targeted Auto-Healing (The Self-Correction Loop)
- **Role:** Rewrites flagged answers without regenerating from scratch.
- **Protocol:**
  - Triggers automatically when `entailment_score < strictness_cutoff`.
  - Constructs a correction prompt highlighting the specific `hallucinated_part`.
  - Forces the generator to excise the hallucination and explicitly state document silence.
  - Runs a second critic verification pass before final delivery.

## 4. Skill: Real-Time Audit Telemetry & Metrics
- **Role:** Formats live latency, pass/fail flags, and score deltas for judge displays.
- **Output:** Emits structured payloads containing unshielded text, NLI score history, and final cleaned text.

## 5. Skill: Token-Level Visual Saliency Diff
- **Role:** Translates critic extractions into side-by-side visual diffs for instantaneous judge inspection.
- **Protocol:**
  - Flags excised hallucinated phrases with high-contrast red strikethroughs (`<span style="text-decoration: line-through;">`).
  - Highlights verified grounded claims and legal silence assertions with emerald green badges.

## 6. Skill: EU AI Act Regulatory Compliance Certification
- **Role:** Generates cryptographically verifiable audit records satisfying EU AI Act Regulation 2024/1689 (Articles 14 & 15).
- **Protocol:**
  - Computes SHA-256 cryptographic digests of retrieved chunk premises, drafts, and finalized answers.
  - Generates immutable serial IDs (`CERT-NLI-EU-XXXXXXXXXX`).
  - Signs every transaction payload with an HMAC-SHA256 digital verification seal.
  - Exports one-click downloadable JSON compliance certificates for enterprise regulatory filings.

## 7. Skill: OpenAI Drop-In Proxy Middleware
- **Role:** Exposes `/v1/chat/completions` and `/v1/models` matching the OpenAI REST specification.
- **Protocol:**
  - Extracts premise context and user inquiries from standard OpenAI message arrays.
  - Transparently audits and self-heals outputs through the NLI engine.
  - Enables zero-code-change drop-in compatibility with LangChain, LlamaIndex, and LiteLLM.

## 8. Skill: Adversarial Red-Team Defense & Stress Testing
- **Role:** Detects and neutralizes malicious prompt injections, authority fabrication baits, and threshold manipulation attacks.
- **Protocol:**
  - Evaluates hypothesis against premise independence; external injection instructions cannot override the NLI Critic's mathematical entailment evaluation.
  - Delivers guarded denials and protects high-stakes compliance thresholds against adversarial deception.
