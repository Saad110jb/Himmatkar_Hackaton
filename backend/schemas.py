from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class NLIVerdict(BaseModel):
    """Pydantic model representing the atomic NLI Critic audit verdict."""
    entailment_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="NLI entailment score ranging from 0.0 (total contradiction) to 1.0 (pure entailment)."
    )
    verdict: str = Field(
        ...,
        description="Categorical NLI classification: 'ENTAILMENT', 'CONTRADICTION', or 'NEUTRAL'."
    )
    hallucinated_part: str = Field(
        default="None",
        description="Flagged unsupported or conflicting claims/entities absent from context premise, or 'None'."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model confidence score in its critique assessment."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief natural language critique justifying the verdict and score."
    )


class TokenDiff(BaseModel):
    """Token-level visual diff metadata for immediate judge inspection."""
    html_diff: str = Field(..., description="Sanitized HTML rendering struck-through red hallucinations and green verified grounded tokens.")
    markdown_diff: str = Field(..., description="Markdown formatted diff with strikethrough for hallucinated spans.")
    flagged_spans: List[str] = Field(default_factory=list, description="Specific phrases extracted and excised by NLI guardrail.")
    replacement_spans: List[str] = Field(default_factory=list, description="Targeted grounded replacements inserted during healing.")


class ComplianceCertificate(BaseModel):
    """Cryptographic EU AI Act Regulatory Compliance Certificate."""
    certificate_id: str = Field(..., description="Unique immutable certificate serial identifier.")
    issued_at: str = Field(..., description="UTC ISO 8601 issuance timestamp.")
    regulatory_framework: str = Field(
        default="EU Artificial Intelligence Act (Regulation 2024/1689)",
        description="Governing regulatory standard."
    )
    articles_covered: List[str] = Field(
        default=[
            "Article 14 - Technical Human Oversight, Traceability & Auditability",
            "Article 15 - Accuracy, Hallucination Prevention & Robustness"
        ],
        description="Statutory EU AI Act requirements certified."
    )
    premise_sha256: str = Field(..., description="Cryptographic SHA-256 digest of input ground-truth premise.")
    initial_draft_sha256: str = Field(..., description="SHA-256 digest of raw unshielded generation.")
    final_output_sha256: str = Field(..., description="SHA-256 digest of delivered production-safe output.")
    critic_model: str = Field(
        default="gemma-4 (Google Generative Language v1beta)",
        description="Underlying NLI critic engine identifier."
    )
    entailment_score: float = Field(..., description="Certified entailment score.")
    verdict: str = Field(..., description="Certified NLI verdict.")
    compliance_status: str = Field(..., description="COMPLIANT_APPROVED or COMPLIANT_AUTO_HEALED.")
    digital_signature: str = Field(..., description="HMAC-SHA256 digital cryptographic signature certifying tamper resistance.")
    telemetry: Dict[str, Any] = Field(default_factory=dict, description="Latency, token lengths, and operational telemetry.")


class AuditRequest(BaseModel):
    """Payload for RAG generation and NLI guardrail audit."""
    context: str = Field(..., description="Ground-truth retrieved premise chunks.")
    question: str = Field(..., description="User query.")
    model: Optional[str] = Field(
        default="gemma-4",
        description="Target model engine: strictly Google Gemma 4 ('gemma-4')."
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional Gemini / Google API key override. If not supplied, uses server environment key."
    )
    temperature: Optional[float] = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the generator stage."
    )
    strictness_cutoff: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Entailment score cutoff threshold. If critic score < strictness_cutoff, self-healing triggers."
    )
    strictness_mode: Optional[str] = Field(
        default="STANDARD",
        description="Risk profile preset: 'ZERO_TOLERANCE' (>=0.85), 'STANDARD' (>=0.70), or 'PERMISSIVE' (>=0.50)."
    )


class AuditResponse(BaseModel):
    """Result of the RAG generation and closed-loop NLI guardrail audit."""
    status: str = Field(..., description="'CLEAN' if initial draft passed guardrail, or 'CORRECTED' if self-healed.")
    initial_draft: str = Field(..., description="Raw, unshielded output from generator LLM.")
    critic_verdict: NLIVerdict = Field(..., description="NLI Critic audit verdict on initial draft.")
    final_output: str = Field(..., description="Shielded, production-safe answer delivered to client.")
    corrected: bool = Field(default=False, description="Flag indicating if the self-correction loop was engaged.")
    second_pass_verdict: Optional[NLIVerdict] = Field(
        default=None,
        description="Optional second-pass NLI audit on the healed output."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Total roundtrip pipeline latency in milliseconds."
    )
    model_used: str = Field(
        default="gemma-4",
        description="Underlying model engine utilized for generation and NLI auditing (strictly Gemma 4)."
    )
    strictness_cutoff: float = Field(
        default=0.70,
        description="Applied entailment threshold cutoff."
    )
    strictness_mode: str = Field(
        default="STANDARD",
        description="Risk profile preset: 'ZERO_TOLERANCE', 'STANDARD', or 'PERMISSIVE'."
    )
    strictness_description: str = Field(
        default="Standard Enterprise Compliance (Cutoff >= 0.70)",
        description="Detailed description of active strictness preset."
    )
    visual_diff: Optional[TokenDiff] = Field(
        default=None,
        description="Token-level visual diff highlighting excised hallucinations and grounded replacements."
    )
    compliance_certificate: Optional[ComplianceCertificate] = Field(
        default=None,
        description="Cryptographically signed EU AI Act compliance certificate."
    )


# ----------------- OpenAI Proxy Models -----------------
class OpenAIChatMessage(BaseModel):
    role: str = Field(..., description="'system', 'user', or 'assistant'")
    content: str = Field(..., description="Text content of the message.")


class OpenAIChatRequest(BaseModel):
    """Standard OpenAI Chat Completions request format for drop-in proxy compatibility."""
    model: Optional[str] = Field(default="gemma-4")
    messages: List[OpenAIChatMessage] = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=0.1)
    strictness_cutoff: Optional[float] = Field(default=0.70)
    api_key: Optional[str] = Field(default=None)


class OpenAIChatChoice(BaseModel):
    index: int = 0
    message: OpenAIChatMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    """Standard OpenAI Chat Completions response object."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChatChoice]
    usage: OpenAIUsage
    guardrail_audit: Optional[AuditResponse] = None


# ----------------- Benchmark & Adversarial Models -----------------
class Scenario(BaseModel):
    """Pre-configured judge demo scenario."""
    id: str
    title: str
    category: str
    context: str
    question: str
    expected_flaw: str
    expected_verdict: str


class AdversarialAttackCase(BaseModel):
    """Adversarial prompt-injection and hallucination-bait attack case."""
    id: str
    attack_type: str
    title: str
    context: str
    adversarial_prompt: str
    threat_analysis: str
    expected_defense: str


class BatchTestCase(BaseModel):
    """Evaluation benchmark test case."""
    id: str
    category: str
    title: str
    context: str
    question: str
    expected_behavior: str
    expected_verdict: str


class BatchTestResultItem(BaseModel):
    """Result of a single benchmark test execution."""
    id: str
    category: str
    title: str
    question: str
    initial_draft: str
    initial_score: float
    initial_verdict: str
    hallucinated_part: str
    status: str
    final_output: str
    final_score: Optional[float] = None
    passed: bool
    latency_ms: float
    certificate_id: Optional[str] = None


class BatchTestResponse(BaseModel):
    """Aggregate benchmark evaluation response."""
    total_tests: int
    passed_tests: int
    pass_rate_percentage: float
    average_latency_ms: float
    results: List[BatchTestResultItem]
    summary: Dict[str, Any]
