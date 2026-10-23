import asyncio
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure project root and backend dir are in sys.path regardless of execution directory
_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.schemas import (
        AuditRequest,
        AuditResponse,
        Scenario,
        AdversarialAttackCase,
        BatchTestResponse,
        BatchTestResultItem,
        OpenAIChatRequest,
        OpenAIChatResponse,
        OpenAIChatChoice,
        OpenAIChatMessage,
        OpenAIUsage
    )
    from backend.guardrail import run_rag_guardrail_pipeline
    from backend.scenarios import BENCHMARK_SUITE, DEMO_SCENARIOS, ADVERSARIAL_ATTACK_SUITE
except ImportError:
    from schemas import (
        AuditRequest,
        AuditResponse,
        Scenario,
        AdversarialAttackCase,
        BatchTestResponse,
        BatchTestResultItem,
        OpenAIChatRequest,
        OpenAIChatResponse,
        OpenAIChatChoice,
        OpenAIChatMessage,
        OpenAIUsage
    )
    from guardrail import run_rag_guardrail_pipeline
    from scenarios import BENCHMARK_SUITE, DEMO_SCENARIOS, ADVERSARIAL_ATTACK_SUITE

app = FastAPI(
    title="RAG Hallucination Guardrail & NLI Detector API",
    description="Enterprise closed-loop LLM guardrail system detecting and eliminating parametric hallucination with OpenAI-compatible proxy and EU AI Act compliance.",
    version="1.1.0"
)

# Enable CORS for frontend and cross-origin tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint confirming API operational status."""
    return {
        "status": "healthy",
        "service": "RAG Hallucination Guardrail & NLI Detector",
        "version": "1.1.0",
        "engine": "Google Gemma 4 (gemma-4)",
        "features": [
            "Gemma 4 Atomic NLI Critic",
            "Targeted Token Excision",
            "Token-Level Visual Saliency Diff",
            "Cryptographic EU AI Act Certification",
            "OpenAI Drop-In Proxy Middleware (/v1/chat/completions)",
            "Dynamic Strictness Presets",
            "Adversarial Red-Team Defense Suite"
        ]
    }


@app.get("/api/scenarios", response_model=List[Scenario], tags=["Audit"])
async def list_scenarios():
    """Returns curated high-stakes benchmark scenarios for judge demos and UI dropdowns."""
    return DEMO_SCENARIOS


@app.get("/api/adversarial-cases", response_model=List[AdversarialAttackCase], tags=["Adversarial"])
async def list_adversarial_cases():
    """Returns pre-built adversarial red-team prompt-injection and hallucination-bait attack cases."""
    return ADVERSARIAL_ATTACK_SUITE


@app.post("/api/audit", response_model=AuditResponse, tags=["Audit"])
@app.post("/api/rag-audit", response_model=AuditResponse, tags=["Audit"])
async def audit_and_generate(req: AuditRequest) -> AuditResponse:
    """
    Executes the full closed-loop RAG Hallucination Guardrail:
    1. Generator LLM synthesizes draft from context & query.
    2. NLI Critic evaluates entailment score (0.0 to 1.0) and flags hallucinated tokens.
    3. Self-healing loop executes if entailment score < strictness_cutoff (default 0.70).
    4. Delivers audited, production-safe output with visual diff and EU AI Act compliance certificate.
    """
    if not req.context.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Context premise cannot be empty."
        )
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User question cannot be empty."
        )

    try:
        response = await run_rag_guardrail_pipeline(req)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Guardrail pipeline processing error: {str(e)}"
        )


# ================== OPENAI COMPATIBLE PROXY ENDPOINTS ==================
@app.get("/v1/models", tags=["OpenAI Proxy"])
async def list_openai_models():
    """OpenAI-compatible models catalog for LangChain / LlamaIndex / OpenAI SDK."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma-4",
                "object": "model",
                "created": 1700000000,
                "owned_by": "google",
                "permission": [],
                "root": "gemma-4"
            },
            {
                "id": "gemma-4-guardrail-protected",
                "object": "model",
                "created": 1700000000,
                "owned_by": "guardrail-nli",
                "permission": [],
                "root": "gemma-4-guardrail-protected"
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=OpenAIChatResponse, tags=["OpenAI Proxy"])
async def openai_chat_completions(req: OpenAIChatRequest) -> OpenAIChatResponse:
    """
    Drop-in OpenAI-compatible chat completions proxy route powered by Gemma 4.
    Allows existing RAG applications (LangChain, LlamaIndex, LiteLLM) to route through
    the NLI Hallucination Guardrail simply by updating their base_url:
    
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    # Parse Context and Question from message history
    system_prompts = [m.content for m in req.messages if m.role == "system"]
    user_messages = [m.content for m in req.messages if m.role == "user"]
    last_user_msg = user_messages[-1] if user_messages else ""

    context = ""
    question = last_user_msg

    # Check if context was passed in system prompt or within user message format
    if system_prompts:
        context = "\n".join(system_prompts)
    
    # Check if user message format has embedded "Context: ... Question: ..."
    ctx_match = re.search(r"Context:\s*(.*?)(?:Question:|$)", last_user_msg, re.DOTALL | re.IGNORECASE)
    q_match = re.search(r"Question:\s*(.*)", last_user_msg, re.DOTALL | re.IGNORECASE)
    
    if ctx_match:
        context = ctx_match.group(1).strip()
    if q_match:
        question = q_match.group(1).strip()

    if not context:
        # Fallback to standard context if query is isolated
        context = "SMEDA Regulatory Memo (Sec 4.2): Small and Medium Enterprises (SMEs) with turnover below PKR 10 Million are exempt from routine audit filing."

    audit_req = AuditRequest(
        context=context,
        question=question,
        model="gemma-4",
        temperature=req.temperature or 0.1,
        strictness_cutoff=req.strictness_cutoff or 0.70,
        api_key=req.api_key
    )

    audit_res = await run_rag_guardrail_pipeline(audit_req)

    completion_id = f"chatcmpl-guardrail-{uuid.uuid4().hex[:12]}"
    prompt_tokens = len(context.split()) + len(question.split())
    completion_tokens = len(audit_res.final_output.split())

    return OpenAIChatResponse(
        id=completion_id,
        object="chat.completion",
        created=int(time.time()),
        model="gemma-4",
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatMessage(
                    role="assistant",
                    content=audit_res.final_output
                ),
                finish_reason="stop"
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        ),
        guardrail_audit=audit_res
    )


# ================== BENCHMARK EVALUATION ENDPOINT ==================
@app.get("/api/batch-test", response_model=BatchTestResponse, tags=["Benchmark"])
@app.post("/api/batch-test", response_model=BatchTestResponse, tags=["Benchmark"])
async def run_batch_benchmark(
    api_key: Optional[str] = None,
    strictness_cutoff: float = 0.70
) -> BatchTestResponse:
    """
    Executes the 10-case evaluation benchmark containing trap and grounded queries.
    Audits accuracy, hallucination detection rate, self-correction efficacy, and latency metrics.
    """
    results: List[BatchTestResultItem] = []
    total = len(BENCHMARK_SUITE)
    passed_count = 0
    total_latency = 0.0

    for test_case in BENCHMARK_SUITE:
        start_time = time.perf_counter()
        req = AuditRequest(
            context=test_case.context,
            question=test_case.question,
            api_key=api_key,
            temperature=0.1,
            strictness_cutoff=strictness_cutoff
        )
        
        try:
            audit_res = await run_rag_guardrail_pipeline(req)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            total_latency += elapsed_ms

            # Test passes if:
            # - For Trap Queries: Guardrail successfully detects hallucination (score < strictness_cutoff) and executes correction
            # - For Grounded Queries: Entailment score >= strictness_cutoff and draft is accepted clean
            is_trap = "TRAP" in test_case.id
            if is_trap:
                passed = audit_res.critic_verdict.entailment_score < strictness_cutoff and audit_res.corrected
            else:
                passed = audit_res.critic_verdict.entailment_score >= strictness_cutoff and not audit_res.corrected

            if passed:
                passed_count += 1

            results.append(
                BatchTestResultItem(
                    id=test_case.id,
                    category=test_case.category,
                    title=test_case.title,
                    question=test_case.question,
                    initial_draft=audit_res.initial_draft,
                    initial_score=audit_res.critic_verdict.entailment_score,
                    initial_verdict=audit_res.critic_verdict.verdict,
                    hallucinated_part=audit_res.critic_verdict.hallucinated_part,
                    status=audit_res.status,
                    final_output=audit_res.final_output,
                    final_score=audit_res.second_pass_verdict.entailment_score if audit_res.second_pass_verdict else audit_res.critic_verdict.entailment_score,
                    passed=passed,
                    latency_ms=round(elapsed_ms, 2),
                    certificate_id=audit_res.compliance_certificate.certificate_id if audit_res.compliance_certificate else None
                )
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            results.append(
                BatchTestResultItem(
                    id=test_case.id,
                    category=test_case.category,
                    title=test_case.title,
                    question=test_case.question,
                    initial_draft="Execution Error",
                    initial_score=0.0,
                    initial_verdict="ERROR",
                    hallucinated_part=str(e),
                    status="FAILED",
                    final_output=f"Error executing test: {str(e)}",
                    passed=False,
                    latency_ms=round(elapsed_ms, 2)
                )
            )

    pass_rate = round((passed_count / total) * 100.0, 1) if total > 0 else 0.0
    avg_latency = round(total_latency / total, 2) if total > 0 else 0.0

    return BatchTestResponse(
        total_tests=total,
        passed_tests=passed_count,
        pass_rate_percentage=pass_rate,
        average_latency_ms=avg_latency,
        results=results,
        summary={
            "trap_queries_count": sum(1 for t in BENCHMARK_SUITE if "TRAP" in t.id),
            "grounded_queries_count": sum(1 for t in BENCHMARK_SUITE if "GROUNDED" in t.id),
            "guardrail_efficacy": f"{pass_rate}%",
            "strictness_cutoff": strictness_cutoff,
            "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
