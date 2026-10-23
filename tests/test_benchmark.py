import asyncio
import pytest
import httpx
from backend.main import app
from backend.schemas import AuditRequest, NLIVerdict
from backend.guardrail import (
    clean_json_text,
    parse_nli_verdict,
    generate_token_visual_diff,
    generate_compliance_certificate,
    run_rag_guardrail_pipeline
)
from backend.scenarios import BENCHMARK_SUITE


class TestUnitSchemasAndParsing:
    """Unit test suite verifying Pydantic schema validation and parser resilience."""

    def test_nli_verdict_schema_valid(self):
        verdict = NLIVerdict(
            entailment_score=0.92,
            verdict="ENTAILMENT",
            hallucinated_part="None",
            confidence=0.98,
            reasoning="Grounded in premise"
        )
        assert verdict.entailment_score == 0.92
        assert verdict.verdict == "ENTAILMENT"
        assert verdict.hallucinated_part == "None"

    def test_nli_verdict_score_clamping(self):
        with pytest.raises(Exception):
            NLIVerdict(
                entailment_score=1.5,  # Out of bounds (>1.0)
                verdict="ENTAILMENT",
                hallucinated_part="None"
            )

    def test_clean_json_text_markdown_stripping(self):
        raw_input = "```json\n{\"entailment_score\": 0.15, \"verdict\": \"CONTRADICTION\", \"hallucinated_part\": \"Form C-3\"}\n```"
        cleaned = clean_json_text(raw_input)
        assert cleaned.startswith("{")
        assert cleaned.endswith("}")

    def test_parse_nli_verdict_resilience(self):
        raw_input = "```json\n{\"entailment_score\": 0.18, \"verdict\": \"contradiction\", \"hallucinated_part\": \"Form C-3 and 15% tax\", \"confidence\": 0.95}\n```"
        verdict = parse_nli_verdict(raw_input)
        assert verdict.entailment_score == 0.18
        assert verdict.verdict == "CONTRADICTION"
        assert "Form C-3" in verdict.hallucinated_part

    def test_token_visual_diff_generation(self):
        diff = generate_token_visual_diff(
            initial_draft="The applicable tax is 15% with Form C-3.",
            final_output="Tax rates are not available in provided documents.",
            hallucinated_part="Form C-3 and 15%",
            corrected=True
        )
        assert "line-through" in diff.html_diff
        assert "15%" in diff.html_diff
        assert len(diff.flagged_spans) > 0

    def test_compliance_certificate_generation(self):
        verdict = NLIVerdict(
            entailment_score=0.15,
            verdict="CONTRADICTION",
            hallucinated_part="Form C-3",
            confidence=0.95
        )
        cert = generate_compliance_certificate(
            context="SMEs under 10M exempt from audit.",
            initial_draft="Draft with Form C-3.",
            final_output="Information not present in context.",
            verdict=verdict,
            status="CORRECTED",
            latency_ms=450.0
        )
        assert cert.certificate_id.startswith("CERT-NLI-EU-")
        assert cert.compliance_status == "COMPLIANT_AUTO_HEALED"
        assert len(cert.premise_sha256) == 64
        assert len(cert.final_output_sha256) == 64
        assert len(cert.digital_signature) == 64
        assert any("Article 14" in a for a in cert.articles_covered)
        assert any("Article 15" in a for a in cert.articles_covered)


class TestGuardrailCorePipeline:
    """Test suite verifying end-to-end NLI guardrail stages and self-correction."""

    def test_trap_query_1_triggers_correction(self):
        """Trap Query 1: Premise covers Sec 4 turnover limits but query asks for tax rate/form."""
        async def _run():
            req = AuditRequest(
                context="SMEDA Regulatory Memo (Sec 4.2): Small and Medium Enterprises (SMEs) with an annual turnover below PKR 10 Million are exempt from routine audit filing under circular 2024/B. Partnership registrations must be logged with the Registrar of Firms.",
                question="What is the exact income tax rate and form for this partnership?"
            )
            return await run_rag_guardrail_pipeline(req)

        res = asyncio.run(_run())
        assert res.critic_verdict.entailment_score < 0.70
        assert res.critic_verdict.verdict in ("CONTRADICTION", "NEUTRAL")
        assert res.status == "CORRECTED"
        assert res.corrected is True
        assert "not" in res.final_output.lower() or "documents do not contain" in res.final_output.lower()
        assert res.visual_diff is not None
        assert "line-through" in res.visual_diff.html_diff
        assert res.compliance_certificate is not None
        assert res.compliance_certificate.compliance_status == "COMPLIANT_AUTO_HEALED"

    def test_grounded_query_2_passes_clean(self):
        """Grounded Query 2: Query asks for turnover limit which is directly stated in premise."""
        async def _run():
            req = AuditRequest(
                context="SMEDA Regulatory Memo (Sec 4.2): Small and Medium Enterprises (SMEs) with an annual turnover below PKR 10 Million are exempt from routine audit filing under circular 2024/B. Partnership registrations must be logged with the Registrar of Firms.",
                question="What turnover qualifies for an audit exemption?"
            )
            return await run_rag_guardrail_pipeline(req)

        res = asyncio.run(_run())
        assert res.critic_verdict.entailment_score >= 0.70
        assert res.critic_verdict.verdict == "ENTAILMENT"
        assert res.status == "CLEAN"
        assert res.corrected is False
        assert "10 Million" in res.final_output or "10 million" in res.final_output
        assert res.compliance_certificate is not None
        assert res.compliance_certificate.compliance_status == "COMPLIANT_APPROVED"

    def test_dynamic_strictness_presets(self):
        """Verifies that strictness_cutoff dynamically controls self-correction trigger."""
        async def _run_strict():
            # Zero-tolerance mode: cutoff at 0.85
            req = AuditRequest(
                context="SMEDA Regulatory Memo (Sec 4.2): Small and Medium Enterprises (SMEs) with turnover below PKR 10 Million are exempt from routine audit filing.",
                question="What is the exact turnover threshold?",
                strictness_cutoff=0.85
            )
            return await run_rag_guardrail_pipeline(req)

        res = asyncio.run(_run_strict())
        assert res.strictness_cutoff == 0.85
        assert res.critic_verdict.entailment_score >= 0.85


class TestAll10BenchmarkCases:
    """Complete evaluation suite running all 10 high-stakes benchmark queries."""

    @pytest.mark.parametrize("test_case", BENCHMARK_SUITE, ids=lambda c: c.id)
    def test_benchmark_case_execution(self, test_case):
        async def _run():
            req = AuditRequest(
                context=test_case.context,
                question=test_case.question,
                temperature=0.1
            )
            return await run_rag_guardrail_pipeline(req)

        res = asyncio.run(_run())

        if "TRAP" in test_case.id:
            assert res.critic_verdict.entailment_score < 0.70, f"Trap {test_case.id} failed to trigger critic flag"
            assert res.corrected is True
            assert res.status == "CORRECTED"
            assert res.compliance_certificate.compliance_status == "COMPLIANT_AUTO_HEALED"
        else:
            assert res.critic_verdict.entailment_score >= 0.70, f"Grounded {test_case.id} failed entailment check"
            assert res.corrected is False
            assert res.status == "CLEAN"
            assert res.compliance_certificate.compliance_status == "COMPLIANT_APPROVED"


class TestFastAPIEndpoints:
    """Integration test suite verifying REST API contracts."""

    def test_health_check(self):
        async def _run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/health")

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_scenarios_endpoint(self):
        async def _run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/api/scenarios")

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 10

    def test_adversarial_cases_endpoint(self):
        async def _run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/api/adversarial-cases")

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 4
        assert any(c["id"] == "ATTACK-01" for c in data)

    def test_api_audit_endpoint(self):
        async def _run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                payload = {
                    "context": "SMEs with turnover below PKR 10 Million are exempt from audit.",
                    "question": "What is the turnover threshold?"
                }
                return await client.post("/api/audit", json=payload)

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        data = resp.json()
        assert "critic_verdict" in data
        assert "final_output" in data
        assert "visual_diff" in data
        assert "compliance_certificate" in data
        assert data["compliance_certificate"]["certificate_id"].startswith("CERT-NLI-EU-")

    def test_openai_models_endpoint(self):
        async def _run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/v1/models")

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert any(m["id"] == "gemma-4" for m in data["data"])

    def test_openai_chat_completions_proxy(self):
        async def _run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                payload = {
                    "model": "gemma-4",
                    "messages": [
                        {
                            "role": "system",
                            "content": "SMEDA Memo (Sec 4.2): SMEs with turnover below PKR 10 Million are exempt from audit."
                        },
                        {
                            "role": "user",
                            "content": "What is the turnover threshold?"
                        }
                    ]
                }
                return await client.post("/v1/chat/completions", json=payload)

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "10 Million" in data["choices"][0]["message"]["content"] or "10 million" in data["choices"][0]["message"]["content"]
        assert data["model"] == "gemma-4"
        assert "guardrail_audit" in data

    def test_batch_test_endpoint(self):
        async def _run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/api/batch-test")

        resp = asyncio.run(_run())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tests"] == 10
        assert data["passed_tests"] == 10
        assert data["pass_rate_percentage"] == 100.0
        assert all(item.get("certificate_id") for item in data["results"])
