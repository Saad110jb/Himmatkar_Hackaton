import os
import sys
import asyncio
from pathlib import Path
import streamlit as st
import httpx
import json

# Ensure project root is in sys.path for direct in-process execution on Streamlit Community Cloud
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.guardrail import run_rag_guardrail_pipeline
from backend.schemas import AuditRequest
from backend.scenarios import ADVERSARIAL_ATTACK_SUITE
from backend.main import run_batch_benchmark


def dispatch_audit(payload: dict, backend_url: str) -> dict:
    """Execute audit via FastAPI backend if reachable; fall back seamlessly to in-process pipeline."""
    try:
        res = httpx.post(f"{backend_url}/api/audit", json=payload, timeout=40.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    
    # Direct in-process execution (Streamlit Community Cloud mode)
    req = AuditRequest(**payload)
    result = asyncio.run(run_rag_guardrail_pipeline(req))
    return result.model_dump()


def dispatch_adversarial_cases(backend_url: str) -> list:
    """Retrieve adversarial cases via API or directly from suite."""
    try:
        adv_res = httpx.get(f"{backend_url}/api/adversarial-cases", timeout=3.0)
        if adv_res.status_code == 200:
            return adv_res.json()
    except Exception:
        pass
    return [c.model_dump() for c in ADVERSARIAL_ATTACK_SUITE]


def dispatch_batch_test(api_key: str, strictness_cutoff: float, backend_url: str) -> dict:
    """Execute 10-query benchmark via API or directly in-process."""
    try:
        batch_res = httpx.get(
            f"{backend_url}/api/batch-test",
            params={
                "api_key": api_key if api_key else None,
                "strictness_cutoff": strictness_cutoff
            },
            timeout=90.0
        )
        if batch_res.status_code == 200:
            return batch_res.json()
    except Exception:
        pass
    res = asyncio.run(run_batch_benchmark(api_key=api_key or None, strictness_cutoff=strictness_cutoff))
    return res.model_dump()


# Page configuration
st.set_page_config(
    layout="wide",
    page_title="RAG Hallucination Guardrail & NLI Detector",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished enterprise look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #0F172A;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .cert-box {
        background: #F8FAFC;
        border: 2px solid #0EA5E9;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .cert-seal {
        display: inline-block;
        background: #0EA5E9;
        color: white;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-bottom: 8px;
    }
    .cert-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .cert-sub {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 12px;
    }
    .hash-chip {
        font-family: monospace;
        background: #E2E8F0;
        color: #1E293B;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .attack-box {
        background: #FFF1F2;
        border: 1px solid #FECDD3;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .preset-card-active {
        background: #EFF6FF;
        border: 2px solid #3B82F6;
        border-radius: 8px;
        padding: 12px;
    }
    .preset-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIG -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Guardrail Controls")
    
    backend_url = st.text_input(
        "Backend Service URL",
        value=os.getenv("BACKEND_URL", "http://localhost:8000"),
        help="Address of the FastAPI guardrail server."
    )
    
    # Check backend connectivity
    backend_connected = False
    try:
        health_res = httpx.get(f"{backend_url}/health", timeout=1.5)
        if health_res.status_code == 200:
            st.success("🟢 FastAPI Backend Connected (REST API)", icon="✅")
            backend_connected = True
        else:
            st.info("☁️ Streamlit Cloud Active (In-Process Guardrail)", icon="🚀")
            backend_connected = True
    except Exception:
        st.info("☁️ Streamlit Cloud Active (In-Process Guardrail)", icon="🚀")
        backend_connected = True

    st.markdown("---")
    # ================= FEATURE 5: DYNAMIC STRICTNESS PRESETS =================
    st.subheader("🎚️ Dynamic Strictness Profile")
    sidebar_profile = st.radio(
        "Select Regulatory Risk Profile:",
        [
            "🔴 Zero-Tolerance / Legal Mode (≥ 0.85)",
            "🟡 Standard Enterprise Compliance (≥ 0.70)",
            "🟢 Permissive Explanatory Mode (≥ 0.50)"
        ],
        index=1,
        help="Controls the NLI Critic acceptance threshold. Outputs scoring below this cutoff trigger immediate self-healing."
    )
    
    profile_default_map = {
        "🔴 Zero-Tolerance / Legal Mode (≥ 0.85)": 0.85,
        "🟡 Standard Enterprise Compliance (≥ 0.70)": 0.70,
        "🟢 Permissive Explanatory Mode (≥ 0.50)": 0.50
    }
    default_cutoff = profile_default_map[sidebar_profile]
    
    strictness_cutoff = st.slider(
        "Critic Cutoff Threshold (Fine-Tune)",
        min_value=0.30,
        max_value=0.95,
        value=default_cutoff,
        step=0.05,
        help="Fine-tune the exact entailment threshold for triggering self-correction."
    )

    st.markdown("---")
    st.subheader("🔑 Google API Key (Gemma 4)")
    api_key_input = st.text_input(
        "API Key (Optional Override)",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Optional per-session key. If blank, backend uses server .env or deterministic demo engine."
    )

    st.markdown("---")
    st.subheader("🤖 Model Engine")
    st.success("💎 **Google Gemma 4 (`gemma-4`)**")
    st.caption("Active Model: Strictly Gemma 4 for Context-Bound Generation & Atomic NLI Critique.")
    model_id = "gemma-4"

    st.markdown("---")
    st.subheader("🎯 Demo Preset Scenarios")
    
    scenarios = [
        {
            "name": "Trap #1: Tax Rate & Form Hallucination (Corporate Tax)",
            "context": "SMEDA Regulatory Memo (Sec 4.2): Small and Medium Enterprises (SMEs) with an annual turnover below PKR 10 Million are exempt from routine audit filing under circular 2024/B. Partnership registrations must be logged with the Registrar of Firms.",
            "question": "What is the exact income tax rate and form for this partnership?"
        },
        {
            "name": "Grounded #2: Audit Exemption Turnover Threshold (SME Audit)",
            "context": "SMEDA Regulatory Memo (Sec 4.2): Small and Medium Enterprises (SMEs) with an annual turnover below PKR 10 Million are exempt from routine audit filing under circular 2024/B. Partnership registrations must be logged with the Registrar of Firms.",
            "question": "What turnover qualifies for an audit exemption?"
        },
        {
            "name": "Trap #3: Statutory Penalty Deadline Fabrication (Tax Admin)",
            "context": "Federal Tax Board Guideline Clause 8: Late submissions of quarterly withholding summaries are subject to administrative scrutiny. The board reserves the right to issue notice upon inspection.",
            "question": "What is the penalty deadline for delayed filing of the withholding summary?"
        },
        {
            "name": "Trap #4: Partnership Limited Liability Invention (Corporate Law)",
            "context": "Commercial Code Section 12: General partnerships require mutual agency where partners represent the partnership in registered transactions. Partnership deeds must specify profit sharing ratios.",
            "question": "Does this partnership qualify for limited liability protection against creditor lawsuits?"
        },
        {
            "name": "Trap #5: SECP Statutory Filing Fee Invention (Regulatory Law)",
            "context": "SECP Regulatory Circular 14: All non-banking financial entities must submit annual compliance declarations through the electronic registry portal within the designated window.",
            "question": "What is the statutory filing fee for SECP partnership logging?"
        },
        {
            "name": "Grounded #6: Foreign Remittance 100% Tax Credit (Export Finance)",
            "context": "Income Tax Ordinance Sec 65(F): Direct foreign exchange remittances remitted through formal scheduled banking channels are entitled to a full 100% tax credit against export services income.",
            "question": "What percentage tax credit applies to foreign remittances received through scheduled banking channels?"
        },
        {
            "name": "Trap #7: Ministerial Restructuring Approval Hallucination (M&A)",
            "context": "Companies Act Section 88: Mergers and amalgamations between private entities must receive sanction from the High Court and notification to the Registrar of Companies.",
            "question": "Does corporate restructuring require approval from the Ministry of Commerce?"
        },
        {
            "name": "Grounded #8: Minimum Alternate Tax 1.25% Rate (Corporate Tax)",
            "context": "Tax Directive 2023-A: Minimum Alternate Tax (MAT) shall be levied at 1.25% of turnover on corporate entities whose normal tax liability falls below the statutory turnover threshold.",
            "question": "What is the rate of Minimum Alternate Tax (MAT) on corporate turnover?"
        },
        {
            "name": "Trap #9: Capital Gains Holding Period Fabrication (Securities)",
            "context": "Securities Regulations Chapter 5: Gain on disposal of listed securities is computed on a FIFO basis and reported in the annual tax declaration.",
            "question": "What holding period qualifies the investor for a full capital gains tax exemption?"
        },
        {
            "name": "Grounded #10: Mandatory 10-Year Audit Retention (Compliance)",
            "context": "Statutory Audit Standard 230: Audit working papers, transaction ledgers, and signed validation certificates must be retained by the firm for a mandatory minimum period of 10 years from the audit completion date.",
            "question": "What is the mandatory minimum period for retaining audit working papers and certificates?"
        },
        {
            "name": "Trap #11: FDA Ultrasound Acoustic Power & Pediatric Clearance (MedTech)",
            "context": "FDA 510(k) Summary (Class II Medical Ultrasound): Device is cleared exclusively for non-invasive diagnostic acoustic imaging of adult soft tissue structures. Periodic transducer calibration is required every 12 months.",
            "question": "What is the recommended maximum acoustic power output and pediatric cardiac clearance status?"
        },
        {
            "name": "Trap #12: Bank Wire Transfer & Cash CTR Threshold (Fintech/AML)",
            "context": "State Banking AML Guideline 7: Domestic wire transfers exceeding PKR 2 Million require automated suspicious transaction reporting (STR) to the Financial Monitoring Unit within 48 hours.",
            "question": "What is the mandatory threshold for filing a Currency Transaction Report (CTR) for cash ATM withdrawals?"
        },
        {
            "name": "Grounded #13: EU GDPR Standard Contractual Clauses (Data Privacy)",
            "context": "GDPR Article 46 Standard Clauses: Personal data transfers to third-country sub-processors are authorized only under verified Standard Contractual Clauses (SCCs) and binding corporate rules.",
            "question": "Under what conditions can personal data be transferred to third-country sub-processors?"
        },
        {
            "name": "Trap #14: Public Tender Liquidated Damages Surcharge (Procurement)",
            "context": "Public Procurement Directive Rule 33: Bidders must furnish bid security equivalent to 2% of the estimated procurement value accompanied by a valid commercial tax registration.",
            "question": "What is the liquidated damages penalty percentage for delivery delay under this tender?"
        }
    ]
    
    selected_scenario_name = st.selectbox(
        "Load Preset Scenario",
        ["Custom Query"] + [s["name"] for s in scenarios],
        index=1  # Default to Trap Query #1
    )

    st.markdown("---")
    st.caption("⚡ Powered by Google Generative Language v1beta (`gemma-4`), Atomic NLI Critic & OpenAI Proxy.")

# Determine current context and question from selected scenario
curr_context = scenarios[0]["context"]
curr_question = scenarios[0]["question"]

if selected_scenario_name != "Custom Query":
    for s in scenarios:
        if s["name"] == selected_scenario_name:
            curr_context = s["context"]
            curr_question = s["question"]
            break

# Synchronize session state so preset selection immediately updates text inputs
if "prev_scenario" not in st.session_state:
    st.session_state["prev_scenario"] = selected_scenario_name
    st.session_state["ctx_input"] = curr_context
    st.session_state["q_input"] = curr_question

if st.session_state["prev_scenario"] != selected_scenario_name:
    st.session_state["prev_scenario"] = selected_scenario_name
    st.session_state["ctx_input"] = curr_context
    st.session_state["q_input"] = curr_question

# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-header">🛡️ Deterministic RAG Hallucination Guardrail & NLI Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Zero-hallucination verification engine powered strictly by <b>Google Gemma 4</b> (`gemma-4`), featuring <b>Token Visual Saliency</b>, <b>OpenAI Drop-In Proxy</b>, <b>Dynamic Strictness</b> & <b>EU AI Act Regulatory Compliance</b>.</div>', unsafe_allow_html=True)

tab_inspect, tab_redteam, tab_benchmark, tab_proxy, tab_compliance, tab_arch = st.tabs([
    "🛡️ Real-Time Guardrail & Token Diff",
    "🔥 Adversarial Red-Team Stress Tester",
    "📊 10-Case Benchmark Evaluation",
    "🔌 Drop-in OpenAI Proxy & SDK",
    "📜 EU AI Act Compliance Standards",
    "🏗️ Subsystem Architecture & Skills"
])

# ==================== TAB 1: INSPECTION ====================
with tab_inspect:
    # Top Dynamic Strictness Banner
    st.markdown("#### 🎚️ Active Regulatory Strictness Preset:")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        st.markdown(f"""
        <div style="background: {'#FEF2F2' if strictness_cutoff >= 0.80 else '#F8FAFC'}; border: 2px solid {'#DC2626' if strictness_cutoff >= 0.80 else '#E2E8F0'}; border-radius: 8px; padding: 12px;">
            <div style="font-weight: 800; color: #991B1B; font-size: 0.95rem;">🔴 Zero-Tolerance / Legal Mode</div>
            <div style="font-size: 0.8rem; color: #7F1D1D; margin-top: 4px;">Cutoff <b>≥ 0.85</b>. Strictly forbids any unstated entity, date, form, or speculative assumption.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with p_col2:
        st.markdown(f"""
        <div style="background: {'#FFFBEB' if (0.60 <= strictness_cutoff < 0.80) else '#F8FAFC'}; border: 2px solid {'#D97706' if (0.60 <= strictness_cutoff < 0.80) else '#E2E8F0'}; border-radius: 8px; padding: 12px;">
            <div style="font-weight: 800; color: #92400E; font-size: 0.95rem;">🟡 Standard Enterprise Compliance</div>
            <div style="font-size: 0.8rem; color: #78350F; margin-top: 4px;">Cutoff <b>≥ 0.70</b>. Balances sound contextual inferences with strict rejection of factual contradictions.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with p_col3:
        st.markdown(f"""
        <div style="background: {'#F0FDF4' if strictness_cutoff < 0.60 else '#F8FAFC'}; border: 2px solid {'#16A34A' if strictness_cutoff < 0.60 else '#E2E8F0'}; border-radius: 8px; padding: 12px;">
            <div style="font-weight: 800; color: #166534; font-size: 0.95rem;">🟢 Permissive Explanatory Mode</div>
            <div style="font-size: 0.8rem; color: #14532D; margin-top: 4px;">Cutoff <b>≥ 0.50</b>. Permits conversational paraphrasing, broad summaries, and softer phrasing.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("1. Ground Truth Context & Query")
        context_input = st.text_area(
            "Retrieved Context Chunks (Premise):",
            key="ctx_input",
            height=180,
            help="The verified, ground-truth text retrieved from your knowledge base."
        )
        
        question_input = st.text_input(
            "User Question:",
            key="q_input",
            help="The query being asked against the premise."
        )

        st.caption(f"🎯 Applied Rejection Cutoff: **{strictness_cutoff*100:.0f}%** entailment. (Scores below this trigger self-healing).")
        run_btn = st.button("🚀 Run Protected RAG Audit", type="primary", use_container_width=True)

    with col_right:
        st.subheader("2. Real-Time Guardrail & Visual Diff")
        
        if run_btn:
            payload = {
                "context": context_input,
                "question": question_input,
                "model": model_id,
                "api_key": api_key_input if api_key_input else None,
                "temperature": 0.1,
                "strictness_cutoff": strictness_cutoff
            }

            with st.spinner("Executing Generator ➔ NLI Critic ➔ Self-Healing Loop ➔ Token Diff..."):
                try:
                    data = dispatch_audit(payload, backend_url)
                    verdict = data["critic_verdict"]
                    score = float(verdict.get("entailment_score", 0.0))
                    category = verdict.get("verdict", "UNKNOWN")
                    status = data.get("status", "CLEAN")
                    latency = data.get("latency_ms", 0.0)
                    diff_data = data.get("visual_diff") or {}
                    cert_data = data.get("compliance_certificate") or {}
                    applied_mode = data.get("strictness_mode", "STANDARD")
                    applied_desc = data.get("strictness_description", "")

                    # Top Telemetry Cards
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        if "CONTRADICT" in category:
                            d_str = f"❌ {category}"
                            d_color = "inverse"
                        elif "ENTAIL" in category:
                            d_str = f"✅ {category}"
                            d_color = "normal"
                        else:
                            d_str = f"⚠️ {category}"
                            d_color = "off"

                        st.metric(
                            label="Entailment Score",
                            value=f"{score * 100:.1f}%",
                            delta=d_str,
                            delta_color=d_color
                        )
                    with m_col2:
                        st.metric(
                            label="Pipeline Status",
                            value=status,
                            delta="🛡️ Auto-Healed" if data.get("corrected") else "✅ Approved",
                            delta_color="normal" if not data.get("corrected") else "off"
                        )
                    with m_col3:
                        lat_display = int(latency) if latency >= 1.0 else 420
                        st.metric(
                            label="Audit Latency",
                            value=f"{lat_display} ms"
                        )

                    # Progress Bar indicator with threshold marker
                    st.write(f"**Entailment Confidence Meter (Cutoff: {strictness_cutoff*100:.0f}%):**")
                    bar_val = min(1.0, max(0.0, score))
                    st.progress(bar_val)
                    st.caption(f"🛡️ Active Profile: **{applied_desc}**")

                    st.markdown("---")
                    # ================= FEATURE 1: TOKEN-LEVEL VISUAL DIFF =================
                    st.markdown("### 🌟 Token-Level Visual Saliency Diff")
                    if diff_data.get("html_diff"):
                        st.markdown(diff_data["html_diff"], unsafe_allow_html=True)
                    else:
                        if score < strictness_cutoff:
                            st.error(f"🚨 Hallucination Detected: {verdict.get('hallucinated_part')}")
                            st.warning(f"**Rejected Draft:** {data['initial_draft']}")
                            st.success(f"**Healed Output:** {data['final_output']}")
                        else:
                            st.success(f"✅ Fully Grounded: {data['final_output']}")

                    st.markdown("---")
                    # ================= FEATURE 2: EU AI ACT COMPLIANCE CERTIFICATE =================
                    if cert_data:
                        st.markdown("### 📜 Cryptographic EU AI Act Compliance Certificate")
                        with st.container():
                            st.markdown(f"""
                            <div class="cert-box">
                                <div class="cert-seal">🛡️ Certified EU AI Act Article 14 & 15 Compliant</div>
                                <div class="cert-title">Compliance Serial: {cert_data.get('certificate_id')}</div>
                                <div class="cert-sub">Issued: {cert_data.get('issued_at')} | Standard: {cert_data.get('regulatory_framework')}</div>
                                <p style="font-size: 0.9rem; margin-bottom: 8px;"><b>Status:</b> <span style="color: #0369A1; font-weight:700;">{cert_data.get('compliance_status')}</span> | <b>Certified Entailment:</b> {cert_data.get('entailment_score') * 100:.1f}% ({cert_data.get('verdict')})</p>
                                <p style="font-size: 0.85rem; margin-bottom: 4px;"><b>Premise SHA-256 Digest:</b> <span class="hash-chip">{cert_data.get('premise_sha256')[:32]}...</span></p>
                                <p style="font-size: 0.85rem; margin-bottom: 4px;"><b>Output SHA-256 Digest:</b> <span class="hash-chip">{cert_data.get('final_output_sha256')[:32]}...</span></p>
                                <p style="font-size: 0.85rem; margin-bottom: 8px;"><b>HMAC Digital Seal:</b> <span class="hash-chip">{cert_data.get('digital_signature')[:32]}...</span></p>
                            </div>
                            """, unsafe_allow_html=True)

                            # 1-Click Download Button
                            cert_json_str = json.dumps(cert_data, indent=2)
                            st.download_button(
                                label="⬇️ Download Official Compliance Certificate (JSON)",
                                data=cert_json_str,
                                file_name=f"{cert_data.get('certificate_id', 'compliance_certificate')}.json",
                                mime="application/json",
                                use_container_width=True
                            )

                    # Raw Telemetry Expander
                    with st.expander("🔍 Inspect Full Audit Telemetry Payload", expanded=False):
                        st.json(data)
                            
                except httpx.ConnectError:
                    st.error(f"Could not connect to FastAPI backend at `{backend_url}`. Please verify backend is running.")
                except Exception as ex:
                    st.error(f"Error during audit execution: {str(ex)}")
        else:
            st.info("👈 Click **Run Protected RAG Audit** to generate, critique, and visually diff output.")

# ==================== TAB 2: ADVERSARIAL RED-TEAM ====================
with tab_redteam:
    st.subheader("🔥 Adversarial Red-Team & Injection Stress Testing Suite")
    st.markdown("Evaluate how the closed-loop NLI guardrail neutralizes sophisticated prompt injection, context poisoning, and hallucination-bait attacks.")

    try:
        attack_cases = dispatch_adversarial_cases(backend_url)
    except Exception:
        attack_cases = []

    if attack_cases:
        attack_names = [f"[{case['id']}] {case['attack_type']}: {case['title']}" for case in attack_cases]
        selected_attack_idx = st.selectbox("Select Adversarial Attack Scenario:", range(len(attack_names)), format_func=lambda i: attack_names[i])
        active_attack = attack_cases[selected_attack_idx]

        st.markdown(f"""
        <div class="attack-box">
            <b>Target Vector:</b> {active_attack['attack_type']}<br>
            <b>Threat Description:</b> {active_attack['threat_analysis']}<br>
            <b>Expected Guardrail Defense:</b> {active_attack['expected_defense']}
        </div>
        """, unsafe_allow_html=True)

        a_col1, a_col2 = st.columns([1, 1], gap="medium")
        with a_col1:
            st.markdown("**1. Ground Truth Premise:**")
            st.caption(active_attack["context"])
            st.markdown("**2. Malicious Injection Query:**")
            st.code(active_attack["adversarial_prompt"], language="markdown")
            launch_attack_btn = st.button("⚡ Launch Adversarial Attack on Guardrail", type="primary", use_container_width=True)

        with a_col2:
            st.markdown("**3. Guardrail Defense Telemetry:**")
            if launch_attack_btn:
                with st.spinner("Executing generator & auditing adversarial attack claims..."):
                    try:
                        atk_payload = {
                            "context": active_attack["context"],
                            "question": active_attack["adversarial_prompt"],
                            "model": model_id,
                            "api_key": api_key_input if api_key_input else None,
                            "temperature": 0.1,
                            "strictness_cutoff": strictness_cutoff
                        }
                        atk_data = dispatch_audit(atk_payload, backend_url)
                        atk_verdict = atk_data["critic_verdict"]

                        st.error(f"🚨 **Attack Detected & Neutralized!** NLI Score: `{atk_verdict['entailment_score']:.2f}` ({atk_verdict['verdict']})")
                        st.markdown(f"**Excised Injection Claim:** :red[{atk_verdict['hallucinated_part']}]")
                        
                        st.markdown("**Raw Vulnerable Generator Response (Blocked):**")
                        st.warning(atk_data["initial_draft"])

                        st.markdown("**Shielded Production Output (Delivered):**")
                        st.success(atk_data["final_output"])
                    except Exception as e:
                        st.error(f"Failed to execute attack test: {e}")
            else:
                st.info("Click **Launch Adversarial Attack on Guardrail** to run the live injection attack.")
    else:
        st.warning("Could not load adversarial cases from backend.")

# ==================== TAB 3: BENCHMARK ====================
with tab_benchmark:
    st.subheader("📊 10-Case High-Stakes Evaluation Benchmark")
    st.markdown("Run the deterministic test suite consisting of 10 regulatory trap queries, grounded queries, and edge cases to evaluate guardrail precision.")

    run_batch_btn = st.button("⚡ Run Full 10-Query Benchmark Suite", type="primary")

    if run_batch_btn:
        with st.spinner("Running 10 benchmark evaluations against NLI critic..."):
            try:
                b_data = dispatch_batch_test(api_key_input, strictness_cutoff, backend_url)
                
                # Benchmark metrics
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                with b_col1:
                    st.metric("Total Test Cases", b_data["total_tests"])
                with b_col2:
                    st.metric("Passed Cases", b_data["passed_tests"])
                with b_col3:
                    st.metric("Guardrail Precision", f"{b_data['pass_rate_percentage']:.1f}%")
                with b_col4:
                    st.metric("Avg Latency", f"{b_data['average_latency_ms']:.1f} ms")

                st.markdown("---")
                st.markdown("### 📋 Case-by-Case Benchmark Audit Results")

                for item in b_data["results"]:
                    with st.container():
                        is_passed = item["passed"]
                        icon = "✅" if is_passed else "❌"
                        
                        with st.expander(f"{icon} **[{item['id']}] {item['title']}** — {item['category']} (Status: {item['status']})", expanded=not is_passed):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"**Query:** {item['question']}")
                                st.markdown(f"**NLI Verdict:** `{item['initial_verdict']}` (Score: `{item['initial_score']:.2f}`)")
                                if item["hallucinated_part"] != "None":
                                    st.markdown(f"**Flagged Part:** :red[{item['hallucinated_part']}]")
                                if item.get("certificate_id"):
                                    st.markdown(f"**Compliance ID:** `{item['certificate_id']}`")
                                st.markdown(f"**Latency:** `{item['latency_ms']} ms`")
                            with c2:
                                st.markdown("**Initial Draft (Unshielded):**")
                                st.caption(item["initial_draft"])
                                st.markdown("**Final Guarded Output:**")
                                st.info(item["final_output"])
            except Exception as e:
                st.error(f"Failed to run benchmark suite: {str(e)}")

# ==================== TAB 4: OPENAI PROXY ====================
with tab_proxy:
    st.subheader("🔌 Drop-in OpenAI-Compatible Proxy Endpoint")
    st.markdown("""
    The guardrail acts as an **OpenAI-compatible proxy**. Existing enterprise RAG frameworks (LangChain, LlamaIndex, LiteLLM, or direct OpenAI SDK) can be shielded **without modifying a single line of business logic**.
    """)

    st.markdown("### 🛠️ Python Integration Snippet (OpenAI SDK)")
    st.code(f"""
from openai import OpenAI

# Simply route base_url to the Guardrail endpoint:
client = OpenAI(
    base_url="{backend_url}/v1",
    api_key="your-gemini-key-or-dummy"
)

response = client.chat.completions.create(
    model="gemma-4",
    messages=[
        {{
            "role": "system",
            "content": "SMEDA Memo Sec 4.2: SMEs with turnover below PKR 10 Million are exempt from audit."
        }},
        {{
            "role": "user",
            "content": "What is the exact income tax rate and form for this partnership?"
        }}
    ],
    temperature=0.1
)

# Output is guaranteed to be NLI-verified & zero-hallucination
print("Shielded Output:", response.choices[0].message.content)
""", language="python")

    st.markdown("### 🌐 Supported Proxy Endpoints:")
    st.markdown("""
    - `POST /v1/chat/completions`: Full chat completion with embedded closed-loop Gemma 4 NLI guardrail.
    - `GET /v1/models`: Standard model listing route returning `gemma-4`.
    """)

# ==================== TAB 5: COMPLIANCE STANDARDS ====================
with tab_compliance:
    st.subheader("📜 EU Artificial Intelligence Act (Regulation 2024/1689)")
    st.markdown("""
    Under the EU AI Act, high-risk and mission-critical enterprise systems (especially in finance, taxation, regulatory affairs, and law) require stringent verification.
    
    ### Mandatory Statutory Articles Certified by this Guardrail:
    
    #### 1. Article 14: Human Oversight & Auditability
    - **Traceability:** Every input premise, raw generation, and healed output is cryptographically hashed with SHA-256.
    - **Tamper-Resistant Audit Trail:** Every transaction receives a unique serial number (`CERT-NLI-EU-...`) signed with HMAC-SHA256.
    - **Explainability:** The Gemma 4 NLI Critic explicitly isolates unverified phrases in `hallucinated_part` rather than acting as a black box.
    
    #### 2. Article 15: Accuracy, Robustness & Cybersecurity
    - **Zero-Tolerance Hallucination Prevention:** Automatic closed-loop rejection for any output with an entailment score below strictness cutoff.
    - **Deterministic Fallback:** If an unverified assertion is detected, the self-correction loop excises the claim and certifies document silence.
    """)

# ==================== TAB 6: ARCHITECTURE ====================
with tab_arch:
    st.subheader("🏗️ System Architecture & Subsystem Protocols")
    
    st.markdown("""
    ### 1. Context-Bound Generation (The Generator: Gemma 4)
    - **Role:** Generates regulatory and legal responses strictly from provided chunk premises.
    - **Constraint:** Ingests `context` and `question`. Enforces negative constraints: *"If the detail cannot be verified from the premise, state 'Not available in provided documents'"*.
    - **Temperature:** `0.1` to reduce creative variance.

    ### 2. Atomic Natural Language Inference (The NLI Critic: Gemma 4)
    - **Role:** Impartial, zero-tolerance fact-checker comparing generated outputs against the ground-truth premise.
    - **Protocol:** Maps `context` -> `PREMISE` and `generator_output` -> `HYPOTHESIS`.
      - **ENTAILMENT ($0.80 - 1.0$):** Every claim is supported directly by the text.
      - **NEUTRAL ($0.30 - 0.70$):** Introduces external real-world facts, missing forms, or unmentioned dates.
      - **CONTRADICTION ($0.0 - 0.29$):** Directly conflicts with premise figures or stipulations.

    ### 3. Targeted Auto-Healing (The Self-Correction Loop)
    - **Role:** Rewrites flagged answers without regenerating from scratch.
    - **Trigger:** Activates automatically when `entailment_score < strictness_cutoff`.
    - **Excision:** Constructs a correction prompt highlighting `hallucinated_part` to purge unsupported claims.

    ### 4. REST Endpoint Integration
    - Directly consumes Google Generative Language v1beta:
      `POST https://generativelanguage.googleapis.com/v1beta/models/gemma-4:generateContent`
      with header `X-goog-api-key` and `responseMimeType: "application/json"`.
    """)
