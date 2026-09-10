import os
import re
import time
import json
from datetime import datetime
from typing import Any

import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_core.tracers.langchain import LangChainTracer
from langchain_openai import ChatOpenAI
from langsmith import Client


AI_JUDGE_QUESTIONS = {
    "Security": [
        "Does the code expose secrets, tokens, credentials, or unsafe sensitive configuration?",
        "Does the code expose or process PII without clear protection or minimization?",
        "Are there dangerous execution paths such as eval, exec, shell calls, or unsafe deserialization?",
        "Is external or user input validated and handled safely?",
    ],
    "Quality": [
        "Is the code clear, readable, and appropriately structured for its size?",
        "Are responsibilities separated well enough to keep the code maintainable?",
        "Are naming, comments, and documentation sufficient for another developer to work on it safely?",
        "Does the code avoid obvious duplication or unnecessary complexity?",
    ],
    "Reliability": [
        "Does the code appear to handle edge cases, nulls, empty inputs, and failures correctly?",
        "Is error handling appropriate for the behavior shown?",
        "Would the suggested fixes likely reduce regressions without introducing obvious new risks?",
    ],
    "Testing": [
        "Does the review identify the most important tests that should exist for this code?",
        "Are the suggested fixes testable and easy to verify?",
    ],
    "Grounding": [
        "Are the review comments grounded in the provided code rather than generic advice?",
        "Are the suggested fixes specific to the provided code rather than speculative?",
        "Is there evidence of hallucinated assumptions not supported by the code or findings?",
    ],
}


@tool
def static_code_check(code: str) -> str:
    """Run a lightweight static check on a code snippet."""
    issues = []
    if '"""' not in code and "'''" not in code:
        issues.append("No docstring found.")
    if "TODO" in code:
        issues.append("Contains TODO comment(s) left in the code.")
    if code.count("\n") > 40:
        issues.append("Function/file may be too long - consider splitting it.")
    return "; ".join(issues) if issues else "No obvious issues found."


def build_agent(api_key: str, model_name: str, system_prompt: str):
    llm = ChatOpenAI(model=model_name, api_key=api_key)
    return create_agent(model=llm, tools=[static_code_check], system_prompt=system_prompt)


def extract_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    return str(content)


def extract_agent_text(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return str(result)
    return extract_text(messages[-1].content)


def get_token_usage(result: dict) -> dict:
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for msg in result.get("messages", []):
        usage_meta = getattr(msg, "usage_metadata", None) or {}
        response_meta = getattr(msg, "response_metadata", None) or {}
        token_usage = response_meta.get("token_usage", {})
        usage["input_tokens"] += usage_meta.get("input_tokens", 0) or token_usage.get(
            "prompt_tokens", 0
        )
        usage["output_tokens"] += usage_meta.get("output_tokens", 0) or token_usage.get(
            "completion_tokens", 0
        )
        usage["total_tokens"] += usage_meta.get("total_tokens", 0) or token_usage.get(
            "total_tokens", 0
        )
    return usage


def summarize_messages(result: dict) -> list[dict[str, str]]:
    summary = []
    for index, msg in enumerate(result.get("messages", []), start=1):
        msg_type = getattr(msg, "type", type(msg).__name__)
        content = extract_text(getattr(msg, "content", ""))
        summary.append(
            {
                "index": str(index),
                "type": str(msg_type),
                "preview": (content[:200] + "...") if len(content) > 200 else content,
            }
        )
    return summary


def decode_uploaded_file(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def build_langsmith_runtime(
    enabled: bool,
    api_key: str,
    project_name: str,
    endpoint: str,
    metadata: dict[str, Any],
):
    if not enabled:
        return None, None, {}
    client = Client(api_key=api_key, api_url=endpoint)
    tracer = LangChainTracer(project_name=project_name, client=client, metadata=metadata)
    invoke_config = {
        "callbacks": [tracer],
        "tags": ["streamlit", "multi-agent", "code-review"],
        "metadata": metadata,
    }
    return client, tracer, invoke_config


def get_run_id_and_url(tracer: LangChainTracer):
    latest_run = tracer.latest_run
    if latest_run is None:
        return None, None
    run_id = str(latest_run.id)
    try:
        run_url = tracer.get_run_url()
    except Exception:
        run_url = None
    return run_id, run_url


def submit_langsmith_feedback(client: Client, run_id: str, score: int, key: str, comment: str):
    return client.create_feedback(
        run_id=run_id,
        key=key,
        score=float(score),
        comment=comment or None,
        value={"score": score, "comment": comment},
    )


def clamp_score(score: int) -> int:
    return max(1, min(5, score))


def evaluate_quality(snippet: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lower_code = snippet.lower()
    if "todo" in lower_code:
        score -= 1
        reasons.append("TODO markers suggest unfinished behavior.")
    if "append(" in snippet and "for " in snippet:
        reasons.append("Loop-based collection building may deserve simplification review.")
    if len(snippet.splitlines()) > 40:
        score -= 1
        reasons.append("Long snippet size increases review risk.")
    return clamp_score(score), reasons or ["No major quality concerns detected from heuristics."]


def evaluate_security(snippet: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lower_code = snippet.lower()
    risky_terms = ["api_key", "password", "secret", "token", "eval(", "exec("]
    matches = [term for term in risky_terms if term in lower_code]
    if matches:
        score -= min(3, len(matches))
        reasons.append(f"Sensitive or risky constructs found: {', '.join(matches)}.")
    if "subprocess" in lower_code or "os.system" in lower_code:
        score -= 1
        reasons.append("Process execution paths need input sanitization review.")
    return clamp_score(score), reasons or ["No major security concerns detected from heuristics."]


def evaluate_maintainability(snippet: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    if '"""' not in snippet and "'''" not in snippet:
        score -= 1
        reasons.append("Missing docstring or module documentation.")
    if len(snippet.splitlines()) > 40:
        score -= 1
        reasons.append("Large code blocks are harder to maintain.")
    if snippet.count("if ") + snippet.count("for ") + snippet.count("while ") > 8:
        score -= 1
        reasons.append("Control-flow density is relatively high.")
    return clamp_score(score), reasons or ["No major maintainability concerns detected."]


def evaluate_reliability(snippet: str, review_text: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lower_code = snippet.lower()
    lower_review = review_text.lower()
    if "try:" not in snippet and "except" not in snippet:
        score -= 1
        reasons.append("No explicit error handling detected.")
    if "empty input" in lower_code or "todo" in lower_code:
        score -= 1
        reasons.append("Edge-case handling appears incomplete.")
    if "test" not in lower_review:
        reasons.append("Review output does not yet reference test coverage.")
    return clamp_score(score), reasons or ["No major reliability concerns detected."]


def evaluate_documentation(snippet: str, review_text: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lower_review = review_text.lower()
    if '"""' not in snippet and "'''" not in snippet:
        score -= 2
        reasons.append("No docstring detected.")
    if "comment" not in lower_review and "document" not in lower_review:
        reasons.append("Review did not identify any documentation expectations.")
    return clamp_score(score), reasons or ["No major documentation concerns detected."]


def evaluate_performance(snippet: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lower_code = snippet.lower()
    if len(snippet.splitlines()) > 60:
        score -= 1
        reasons.append("Long routines often hide avoidable work.")
    if ".append(" in snippet and "for " in snippet:
        reasons.append("Potential vector for list-comprehension or batching improvement.")
    if lower_code.count("for ") > 2:
        score -= 1
        reasons.append("Multiple loops may need complexity review.")
    return clamp_score(score), reasons or ["No major performance concerns detected."]


def evaluate_readability(snippet: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lines = [line for line in snippet.splitlines() if line.strip()]
    if any(len(line) > 100 for line in lines):
        score -= 1
        reasons.append("Some lines are longer than 100 characters.")
    if len(lines) > 25:
        score -= 1
        reasons.append("Long snippets are harder to scan quickly.")
    if not any(line.strip().startswith("#") for line in lines) and '"""' not in snippet and "'''" not in snippet:
        score -= 1
        reasons.append("No inline comments or docstrings detected.")
    return clamp_score(score), reasons or ["Readability looks acceptable from heuristics."]


def evaluate_testability(snippet: str, review_text: str, fix_text: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lower_review = review_text.lower()
    lower_fix = fix_text.lower()
    if "test" not in lower_review:
        score -= 1
        reasons.append("Review comments did not mention tests.")
    if "test" not in lower_fix:
        score -= 1
        reasons.append("Fix suggestions did not include test ideas.")
    if snippet.count("def ") > 3 or snippet.count("class ") > 1:
        reasons.append("Broader code surface may need multiple test cases.")
    return clamp_score(score), reasons or ["Testability signals look acceptable."]


def evaluate_secret_exposure(snippet: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lower_code = snippet.lower()
    markers = ["api_key", "secret", "token", "password", "private_key"]
    matches = [marker for marker in markers if marker in lower_code]
    if matches:
        score -= min(4, len(matches))
        reasons.append(f"Potential secret-bearing identifiers found: {', '.join(matches)}.")
    if re.search(r"sk-[A-Za-z0-9_-]{20,}", snippet):
        score = min(score, 1)
        reasons.append("String looks like a live API key.")
    return clamp_score(score), reasons or ["No obvious secret exposure detected."]


def evaluate_pii_exposure(snippet: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    pii_hits = []
    if re.search(r"\b[^\s@]+@[^\s@]+\.[A-Za-z]{2,}\b", snippet):
        pii_hits.append("email")
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", snippet):
        pii_hits.append("ssn-like")
    if re.search(r"\b\+?\d[\d\s().-]{8,}\d\b", snippet):
        pii_hits.append("phone-like")
    if re.search(r"\b\d{13,19}\b", snippet):
        pii_hits.append("card/account-like")
    if pii_hits:
        score -= min(4, len(set(pii_hits)))
        reasons.append(f"Potential PII patterns detected: {', '.join(sorted(set(pii_hits)))}.")
    return clamp_score(score), reasons or ["No obvious PII patterns detected."]


def evaluate_input_safety(snippet: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lower_code = snippet.lower()
    if "input(" in lower_code or "request." in lower_code or "argv" in lower_code:
        reasons.append("Code appears to consume user input; validation should be checked.")
    if any(term in lower_code for term in ["eval(", "exec(", "pickle.loads", "yaml.load("]):
        score -= 2
        reasons.append("Unsafe deserialization or dynamic execution pattern detected.")
    if any(term in lower_code for term in ["subprocess", "os.system", "shell=true"]):
        score -= 1
        reasons.append("Command execution path may need sanitization.")
    return clamp_score(score), reasons or ["No obvious input handling risks detected."]


def evaluate_hallucination_risk(snippet: str, review_text: str, fix_text: str) -> tuple[int, list[str]]:
    score = 5
    reasons = []
    lower_context = f"{snippet}\n{review_text}".lower()
    lower_fix = fix_text.lower()
    external_terms = [
        "database",
        "sql",
        "authentication",
        "jwt",
        "docker",
        "kubernetes",
        "redis",
        "aws",
        "s3",
        "microservice",
    ]
    unsupported = [term for term in external_terms if term in lower_fix and term not in lower_context]
    if unsupported:
        score -= min(3, len(unsupported))
        reasons.append(f"Fixes mention concepts not grounded in the input: {', '.join(unsupported[:5])}.")
    if "improved version" not in lower_fix and "fix" not in lower_fix:
        score -= 1
        reasons.append("Fix response may be too generic to verify against the code.")
    return clamp_score(score), reasons or ["Generated suggestions look reasonably grounded in the input."]


def average_scores(scores: dict[str, int]) -> float:
    return round(sum(scores.values()) / len(scores), 2)


def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        newline_index = cleaned.find("\n")
        if newline_index != -1:
            cleaned = cleaned[newline_index + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("AI Judge did not return a valid JSON object.")
    return json.loads(cleaned[start : end + 1])


def build_ai_judge_prompt(language: str, snippet: str, review_text: str, fix_text: str) -> str:
    questions_block = []
    for category, questions in AI_JUDGE_QUESTIONS.items():
        questions_block.append(category + ":")
        questions_block.extend(f"- {question}" for question in questions)

    joined_questions = "\n".join(questions_block)
    return (
        f"Evaluate the following {language} code, the review comments, and the suggested fixes.\n\n"
        "Use only the provided inputs. Do not invent missing architecture, services, or behavior.\n"
        "For each question, return a score from 1 to 5, a verdict of pass/partial/fail, and a brief evidence-based reason.\n"
        "Then compute average scores per category and an overall score.\n"
        "Return JSON only using this schema:\n"
        "{\n"
        '  "overall_score": number,\n'
        '  "summary": "string",\n'
        '  "top_risks": ["string"],\n'
        '  "categories": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "average_score": number,\n'
        '      "questions": [\n'
        "        {\n"
        '          "question": "string",\n'
        '          "score": number,\n'
        '          "verdict": "pass|partial|fail",\n'
        '          "reason": "string"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Questions:\n{joined_questions}\n\n"
        f"Code:\n{snippet}\n\n"
        f"Review comments:\n{review_text}\n\n"
        f"Suggested fixes:\n{fix_text}"
    )


def run_ai_judge(
    api_key: str,
    model_name: str,
    language: str,
    snippet: str,
    review_text: str,
    fix_text: str,
    tracing_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int], str]:
    judge_llm = ChatOpenAI(model=model_name, api_key=api_key)
    response = judge_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an AI Judge for code review quality, security, reliability, and grounding. "
                    "Be strict, evidence-based, and return valid JSON only."
                )
            ),
            HumanMessage(content=build_ai_judge_prompt(language, snippet, review_text, fix_text)),
        ],
        config=tracing_config,
    )
    usage_meta = getattr(response, "usage_metadata", None) or {}
    usage = {
        "input_tokens": usage_meta.get("input_tokens", 0),
        "output_tokens": usage_meta.get("output_tokens", 0),
        "total_tokens": usage_meta.get("total_tokens", 0),
    }
    raw_text = extract_text(response.content)
    return extract_json_payload(raw_text), usage, raw_text


def build_evaluation_metrics(snippet: str, review_text: str, fix_text: str) -> dict[str, Any]:
    evaluators = {
        "quality": lambda: evaluate_quality(snippet),
        "security": lambda: evaluate_security(snippet),
        "maintainability": lambda: evaluate_maintainability(snippet),
        "reliability": lambda: evaluate_reliability(snippet, review_text),
        "documentation": lambda: evaluate_documentation(snippet, review_text),
        "performance": lambda: evaluate_performance(snippet),
    }
    scores = {}
    reasons = {}
    for dimension, evaluator in evaluators.items():
        score, dimension_reasons = evaluator()
        scores[dimension] = score
        reasons[dimension] = dimension_reasons

    if "improved code" in fix_text.lower() or "prioritized" in fix_text.lower():
        scores["quality"] = clamp_score(scores["quality"] + 1)

    quality_subscores = {
        "quality": scores["quality"],
        "readability": evaluate_readability(snippet)[0],
        "maintainability": scores["maintainability"],
        "documentation": scores["documentation"],
        "testability": evaluate_testability(snippet, review_text, fix_text)[0],
        "performance": scores["performance"],
    }
    security_subscores = {
        "security": scores["security"],
        "pii_exposure": evaluate_pii_exposure(snippet)[0],
        "secret_exposure": evaluate_secret_exposure(snippet)[0],
        "input_safety": evaluate_input_safety(snippet)[0],
    }
    ai_review_subscores = {
        "reliability": scores["reliability"],
        "hallucination_risk": evaluate_hallucination_risk(snippet, review_text, fix_text)[0],
        "grounding": evaluate_hallucination_risk(snippet, review_text, fix_text)[0],
        "fix_actionability": evaluate_testability(snippet, review_text, fix_text)[0],
    }

    grouped_reasons = {
        "quality": {
            "quality": reasons["quality"],
            "readability": evaluate_readability(snippet)[1],
            "maintainability": reasons["maintainability"],
            "documentation": reasons["documentation"],
            "testability": evaluate_testability(snippet, review_text, fix_text)[1],
            "performance": reasons["performance"],
        },
        "security": {
            "security": reasons["security"],
            "pii_exposure": evaluate_pii_exposure(snippet)[1],
            "secret_exposure": evaluate_secret_exposure(snippet)[1],
            "input_safety": evaluate_input_safety(snippet)[1],
        },
        "ai_review": {
            "reliability": reasons["reliability"],
            "hallucination_risk": evaluate_hallucination_risk(snippet, review_text, fix_text)[1],
            "grounding": evaluate_hallucination_risk(snippet, review_text, fix_text)[1],
            "fix_actionability": evaluate_testability(snippet, review_text, fix_text)[1],
        },
    }

    category_scores = {
        "quality": average_scores(quality_subscores),
        "security": average_scores(security_subscores),
        "ai_review": average_scores(ai_review_subscores),
    }

    overall = round(sum(scores.values()) / len(scores), 2)
    return {
        "overall": overall,
        "scores": scores,
        "reasons": reasons,
        "categories": {
            "quality": quality_subscores,
            "security": security_subscores,
            "ai_review": ai_review_subscores,
        },
        "category_scores": category_scores,
        "category_reasons": grouped_reasons,
    }


def init_state():
    defaults = {
        "snippet": (
            "def process(data):\n"
            "    # TODO: handle empty input\n"
            "    result = []\n"
            "    for d in data:\n"
            "        result.append(d * 2)\n"
            "    return result\n"
        ),
        "usage_history": [],
        "activity_logs": [],
        "evaluation_logs": [],
        "evaluation_history": [],
        "judge_result": None,
        "judge_history": [],
        "review_output": "",
        "fix_output": "",
        "last_review_run_id": None,
        "last_fix_run_id": None,
        "last_review_run_url": None,
        "last_fix_run_url": None,
        "langsmith_status": None,
        "latest_internals": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_activity_log(entry: dict[str, Any]):
    st.session_state.activity_logs.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **entry,
        }
    )


st.set_page_config(page_title="Developer Agent", layout="wide")
init_state()

st.title("Developer Agent Workspace")
st.caption("Multi-agent review with monitoring, tracing, and internals.")

with st.sidebar:
    st.header("Run Configuration")

    env_openai_key = os.getenv("OPENAI_API_KEY", "")
    openai_key_input = st.text_input(
        "OpenAI API key",
        value="",
        type="password",
        help="If left empty, uses OPENAI_API_KEY from environment.",
    )

    model_option = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "custom"],
        index=0,
    )
    custom_model = st.text_input("Custom model name", value="", disabled=model_option != "custom")
    selected_model = custom_model.strip() if model_option == "custom" else model_option

    language = st.selectbox(
        "Code language",
        options=[
            "Python",
            "JavaScript",
            "TypeScript",
            "Java",
            "C#",
            "C++",
            "Go",
            "Rust",
            "SQL",
            "Other",
        ],
        index=0,
    )

    st.subheader("LangSmith")
    enable_langsmith = st.checkbox("Enable tracing", value=False)
    env_langsmith_key = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_key_input = st.text_input(
        "LangSmith API key",
        value="",
        type="password",
        help="If left empty, uses LANGSMITH_API_KEY from environment.",
        disabled=not enable_langsmith,
    )
    langsmith_project = st.text_input(
        "Project",
        value="developer-agent-streamlit",
        disabled=not enable_langsmith,
    )
    langsmith_endpoint = st.text_input(
        "Endpoint",
        value=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        disabled=not enable_langsmith,
    )

    if st.button("Clear Logs and State", width="stretch"):
        for key in [
            "usage_history",
            "activity_logs",
            "evaluation_logs",
            "evaluation_history",
            "judge_history",
            "review_output",
            "fix_output",
            "last_review_run_id",
            "last_fix_run_id",
            "last_review_run_url",
            "last_fix_run_url",
            "langsmith_status",
            "latest_internals",
            "judge_result",
        ]:
            st.session_state[key] = [] if key.endswith(("history", "logs")) else None
        st.session_state.review_output = ""
        st.session_state.fix_output = ""
        st.rerun()

active_openai_key = openai_key_input.strip() or env_openai_key
active_langsmith_key = langsmith_key_input.strip() or env_langsmith_key

if not active_openai_key:
    st.warning("OpenAI API key is required to run review agents.")
if not selected_model:
    st.warning("Select a model before running.")
if enable_langsmith and not active_langsmith_key:
    st.warning("LangSmith tracing is enabled but API key is missing.")

tab_review, tab_monitoring, tab_evaluation, tab_tracing, tab_flow = st.tabs(
    ["Review", "Monitoring", "Evaluation", "Tracing", "Agentic Flow"]
)

with tab_review:
    st.subheader("Code Input")
    input_mode = st.radio(
        "Input mode",
        options=["Paste snippet", "Upload code file"],
        horizontal=True,
    )

    snippet = ""
    source_label = "paste"

    if input_mode == "Paste snippet":
        snippet = st.text_area(
            "Code snippet",
            value=st.session_state.snippet,
            height=320,
            key="snippet_editor",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload code file",
            type=[
                "py",
                "js",
                "ts",
                "java",
                "cs",
                "cpp",
                "c",
                "go",
                "rs",
                "rb",
                "php",
                "swift",
                "kt",
                "scala",
                "sql",
                "txt",
                "md",
            ],
        )
        if uploaded_file is not None:
            snippet = decode_uploaded_file(uploaded_file)
            source_label = f"upload:{uploaded_file.name}"
            st.text_area("Uploaded preview", value=snippet, height=320, disabled=True)

    if st.button("Run Multi-Agent Review", type="primary", width="stretch"):
        if not snippet.strip():
            st.error("Please paste code or upload a file.")
        elif not active_openai_key:
            st.error("OpenAI API key is missing.")
        elif not selected_model:
            st.error("Model name is required.")
        elif enable_langsmith and not active_langsmith_key:
            st.error("LangSmith API key is required when tracing is enabled.")
        else:
            st.session_state.snippet = snippet
            run_id = len(st.session_state.usage_history) + 1
            start_ts = time.time()
            review_result = {}
            fix_result = {}
            review_run_id = None
            review_run_url = None
            fix_run_id = None
            fix_run_url = None

            review_prompt = (
                "You are a senior code reviewer. "
                f"Analyze this {language} code. "
                "Use static_code_check first, then provide concise review comments with severity, risks, and what to fix."
            )
            fixer_prompt = (
                "You are a senior refactoring engineer. "
                f"Given {language} code and review findings, suggest concrete fixes. "
                "Return: 1) prioritized fixes 2) improved code sample 3) test ideas."
            )

            try:
                base_meta = {
                    "app": "developer_agent_streamlit",
                    "run_id": run_id,
                    "language": language,
                    "model": selected_model,
                    "source": source_label,
                }

                with st.spinner("Agent 1/2 reviewing code..."):
                    _, review_tracer, review_config = build_langsmith_runtime(
                        enabled=enable_langsmith,
                        api_key=active_langsmith_key,
                        project_name=langsmith_project,
                        endpoint=langsmith_endpoint,
                        metadata={**base_meta, "agent_role": "review"},
                    )
                    reviewer_agent = build_agent(active_openai_key, selected_model, review_prompt)
                    review_result = reviewer_agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"Review this {language} code:\n{snippet}",
                                }
                            ]
                        },
                        config=review_config,
                    )
                    review_text = extract_agent_text(review_result)
                    review_usage = get_token_usage(review_result)
                    review_run_id, review_run_url = (
                        get_run_id_and_url(review_tracer) if review_tracer else (None, None)
                    )

                with st.spinner("Agent 2/2 suggesting fixes..."):
                    _, fix_tracer, fix_config = build_langsmith_runtime(
                        enabled=enable_langsmith,
                        api_key=active_langsmith_key,
                        project_name=langsmith_project,
                        endpoint=langsmith_endpoint,
                        metadata={**base_meta, "agent_role": "fix"},
                    )
                    fixer_agent = build_agent(active_openai_key, selected_model, fixer_prompt)
                    fix_result = fixer_agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        f"Code language: {language}\n\n"
                                        f"Original code:\n{snippet}\n\n"
                                        f"Review findings:\n{review_text}\n\n"
                                        "Suggest concrete fixes and provide an improved version."
                                    ),
                                }
                            ]
                        },
                        config=fix_config,
                    )
                    fix_text = extract_agent_text(fix_result)
                    fix_usage = get_token_usage(fix_result)
                    fix_run_id, fix_run_url = (
                        get_run_id_and_url(fix_tracer) if fix_tracer else (None, None)
                    )

                total_usage = {
                    "input_tokens": review_usage["input_tokens"] + fix_usage["input_tokens"],
                    "output_tokens": review_usage["output_tokens"] + fix_usage["output_tokens"],
                    "total_tokens": review_usage["total_tokens"] + fix_usage["total_tokens"],
                }

                st.session_state.review_output = review_text
                st.session_state.fix_output = fix_text
                st.session_state.last_review_run_id = review_run_id
                st.session_state.last_fix_run_id = fix_run_id
                st.session_state.last_review_run_url = review_run_url
                st.session_state.last_fix_run_url = fix_run_url

                st.session_state.usage_history.append(
                    {
                        "run": run_id,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "review_tokens": review_usage["total_tokens"],
                        "fix_tokens": fix_usage["total_tokens"],
                        **total_usage,
                    }
                )

                st.session_state.latest_internals = {
                    "run": run_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_sec": round(time.time() - start_ts, 2),
                    "model": selected_model,
                    "language": language,
                    "source": source_label,
                    "input_chars": len(snippet),
                    "input_lines": len(snippet.splitlines()),
                    "review_prompt": review_prompt,
                    "fixer_prompt": fixer_prompt,
                    "review_usage": review_usage,
                    "fix_usage": fix_usage,
                    "total_usage": total_usage,
                    "review_trace": {"run_id": review_run_id, "url": review_run_url},
                    "fix_trace": {"run_id": fix_run_id, "url": fix_run_url},
                    "review_messages": summarize_messages(review_result),
                    "fix_messages": summarize_messages(fix_result),
                }

                evaluation_metrics = build_evaluation_metrics(snippet, review_text, fix_text)
                st.session_state.latest_internals["evaluation_metrics"] = evaluation_metrics
                st.session_state.judge_result = None
                st.session_state.evaluation_history.append(
                    {
                        "run": run_id,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "overall": evaluation_metrics["overall"],
                        "quality_group": evaluation_metrics["category_scores"]["quality"],
                        "security_group": evaluation_metrics["category_scores"]["security"],
                        "ai_review_group": evaluation_metrics["category_scores"]["ai_review"],
                        "pii_exposure": evaluation_metrics["categories"]["security"]["pii_exposure"],
                        "hallucination_risk": evaluation_metrics["categories"]["ai_review"]["hallucination_risk"],
                        **evaluation_metrics["scores"],
                    }
                )

                add_activity_log(
                    {
                        "run": run_id,
                        "model": selected_model,
                        "language": language,
                        "source": source_label,
                        "status": "success",
                        "tokens": total_usage["total_tokens"],
                        "review_run_id": review_run_id,
                        "fix_run_id": fix_run_id,
                    }
                )
                st.success("Multi-agent review completed.")
            except Exception as exc:
                add_activity_log(
                    {
                        "run": run_id,
                        "model": selected_model,
                        "language": language,
                        "source": source_label,
                        "status": f"failed: {exc}",
                        "tokens": 0,
                    }
                )
                st.error(f"Review failed: {exc}")

    review_col, fix_col = st.columns(2)
    with review_col:
        st.subheader("Review Comments")
        st.write(st.session_state.review_output or "No review output yet.")
    with fix_col:
        st.subheader("Fix Suggestions")
        st.write(st.session_state.fix_output or "No fix output yet.")

with tab_monitoring:
    st.subheader("Token Usage")
    if st.session_state.usage_history:
        latest = st.session_state.usage_history[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Last Input Tokens", latest["input_tokens"])
        m2.metric("Last Output Tokens", latest["output_tokens"])
        m3.metric("Last Total Tokens", latest["total_tokens"])

        st.line_chart(
            {
                "review_tokens": [row["review_tokens"] for row in st.session_state.usage_history],
                "fix_tokens": [row["fix_tokens"] for row in st.session_state.usage_history],
                "total_tokens": [row["total_tokens"] for row in st.session_state.usage_history],
            }
        )
        st.dataframe(st.session_state.usage_history, width="stretch")
    else:
        st.info("No usage data yet.")

    st.subheader("Review Activity Logs")
    if st.session_state.activity_logs:
        st.dataframe(st.session_state.activity_logs, width="stretch")
    else:
        st.info("No activity logs yet.")

with tab_evaluation:
    st.subheader("AI Judge Evaluation")
    st.caption("Question-based evaluation across security, quality, reliability, testing, and grounding.")

    question_tabs = st.tabs(list(AI_JUDGE_QUESTIONS.keys()))
    for index, category in enumerate(AI_JUDGE_QUESTIONS):
        with question_tabs[index]:
            for question_number, question in enumerate(AI_JUDGE_QUESTIONS[category], start=1):
                st.write(f"{question_number}. {question}")

    if st.button("Run AI Judge", width="stretch"):
        if not st.session_state.snippet.strip():
            st.error("No code snippet available for evaluation.")
        elif not st.session_state.review_output or not st.session_state.fix_output:
            st.error("Run the multi-agent review first so the judge can evaluate both review and fixes.")
        elif not active_openai_key:
            st.error("OpenAI API key is missing.")
        else:
            try:
                judge_trace_config = {}
                judge_run_id = None
                judge_run_url = None
                if enable_langsmith and active_langsmith_key:
                    _, judge_tracer, judge_trace_config = build_langsmith_runtime(
                        enabled=True,
                        api_key=active_langsmith_key,
                        project_name=langsmith_project,
                        endpoint=langsmith_endpoint,
                        metadata={
                            "app": "developer_agent_streamlit",
                            "agent_role": "judge",
                            "language": language,
                            "model": selected_model,
                        },
                    )
                else:
                    judge_tracer = None

                with st.spinner("Running AI Judge..."):
                    judge_result, judge_usage, judge_raw = run_ai_judge(
                        api_key=active_openai_key,
                        model_name=selected_model,
                        language=language,
                        snippet=st.session_state.snippet,
                        review_text=st.session_state.review_output,
                        fix_text=st.session_state.fix_output,
                        tracing_config=judge_trace_config,
                    )
                    if judge_tracer:
                        judge_run_id, judge_run_url = get_run_id_and_url(judge_tracer)

                st.session_state.judge_result = judge_result
                st.session_state.latest_internals = st.session_state.latest_internals or {}
                st.session_state.latest_internals["judge_raw"] = judge_raw
                st.session_state.latest_internals["judge_usage"] = judge_usage
                st.session_state.latest_internals["judge_trace"] = {
                    "run_id": judge_run_id,
                    "url": judge_run_url,
                }

                category_scores = {
                    category["name"]: category["average_score"]
                    for category in judge_result.get("categories", [])
                }
                st.session_state.judge_history.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "overall": judge_result.get("overall_score", 0),
                        **category_scores,
                    }
                )
                add_activity_log(
                    {
                        "run": len(st.session_state.usage_history),
                        "model": selected_model,
                        "language": language,
                        "source": "ai_judge",
                        "status": "success",
                        "tokens": judge_usage["total_tokens"],
                    }
                )
                st.success("AI Judge evaluation completed.")
            except Exception as exc:
                st.error(f"AI Judge failed: {exc}")

    latest = st.session_state.latest_internals
    judge_result = st.session_state.judge_result
    if judge_result:
        overview_cols = st.columns(3)
        overview_cols[0].metric("Overall Score", judge_result.get("overall_score", 0))
        overview_cols[1].metric(
            "Top Risk Count", len(judge_result.get("top_risks", []))
        )
        overview_cols[2].metric(
            "Category Count", len(judge_result.get("categories", []))
        )

        category_scores = {
            category["name"]: category["average_score"]
            for category in judge_result.get("categories", [])
        }
        st.subheader("Category Scores")
        st.bar_chart(category_scores, width="stretch")

        st.subheader("Judge Summary")
        st.write(judge_result.get("summary", "No summary returned."))

        if judge_result.get("top_risks"):
            st.subheader("Top Risks")
            for risk in judge_result["top_risks"]:
                st.write(f"- {risk}")

        st.subheader("Question-Level Results")
        result_tabs = st.tabs([category["name"] for category in judge_result.get("categories", [])])
        for index, category in enumerate(judge_result.get("categories", [])):
            with result_tabs[index]:
                st.metric("Category Average", category.get("average_score", 0))
                st.dataframe(category.get("questions", []), width="stretch")
                for question in category.get("questions", []):
                    st.progress(
                        float(question.get("score", 0)) / 5,
                        text=(
                            f"{question.get('verdict', 'partial').title()} - "
                            f"{question.get('question', '')}"
                        ),
                    )

    elif latest and latest.get("evaluation_metrics"):
        metrics = latest["evaluation_metrics"]
        score_cols = st.columns(4)
        score_cols[0].metric("Overall", metrics["overall"])
        score_cols[1].metric("Quality", metrics["scores"]["quality"])
        score_cols[2].metric("Security", metrics["scores"]["security"])
        score_cols[3].metric("Reliability", metrics["scores"]["reliability"])

        more_cols = st.columns(3)
        more_cols[0].metric("Maintainability", metrics["scores"]["maintainability"])
        more_cols[1].metric("Documentation", metrics["scores"]["documentation"])
        more_cols[2].metric("Performance", metrics["scores"]["performance"])

        st.subheader("Category Overview")
        category_cols = st.columns(3)
        category_cols[0].metric("Quality Group", metrics["category_scores"]["quality"])
        category_cols[1].metric("Security Group", metrics["category_scores"]["security"])
        category_cols[2].metric("AI Review Group", metrics["category_scores"]["ai_review"])

        st.bar_chart(metrics["category_scores"], width="stretch")

        quality_col, security_col, ai_col = st.columns(3)
        with quality_col:
            st.markdown("**Quality Metrics**")
            st.bar_chart(metrics["categories"]["quality"], width="stretch")
            for metric_name, score in metrics["categories"]["quality"].items():
                st.progress(score / 5, text=f"{metric_name.replace('_', ' ').title()}: {score}/5")

        with security_col:
            st.markdown("**Security Metrics**")
            st.bar_chart(metrics["categories"]["security"], width="stretch")
            for metric_name, score in metrics["categories"]["security"].items():
                st.progress(score / 5, text=f"{metric_name.replace('_', ' ').title()}: {score}/5")

        with ai_col:
            st.markdown("**AI Review Metrics**")
            st.bar_chart(metrics["categories"]["ai_review"], width="stretch")
            for metric_name, score in metrics["categories"]["ai_review"].items():
                st.progress(score / 5, text=f"{metric_name.replace('_', ' ').title()}: {score}/5")

        st.subheader("Metric Rationale")
        rationale_rows = []
        for dimension, score in metrics["scores"].items():
            rationale_rows.append(
                {
                    "dimension": dimension,
                    "score": score,
                    "notes": " ".join(metrics["reasons"][dimension]),
                }
            )
        st.dataframe(rationale_rows, width="stretch")

        st.subheader("Detailed Evaluation Notes")
        detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Quality", "Security", "AI Review"])
        with detail_tab1:
            st.dataframe(
                [
                    {
                        "metric": key,
                        "score": value,
                        "notes": " ".join(metrics["category_reasons"]["quality"][key]),
                    }
                    for key, value in metrics["categories"]["quality"].items()
                ],
                width="stretch",
            )
        with detail_tab2:
            st.dataframe(
                [
                    {
                        "metric": key,
                        "score": value,
                        "notes": " ".join(metrics["category_reasons"]["security"][key]),
                    }
                    for key, value in metrics["categories"]["security"].items()
                ],
                width="stretch",
            )
        with detail_tab3:
            st.dataframe(
                [
                    {
                        "metric": key,
                        "score": value,
                        "notes": " ".join(metrics["category_reasons"]["ai_review"][key]),
                    }
                    for key, value in metrics["categories"]["ai_review"].items()
                ],
                width="stretch",
            )
    else:
        st.info("No automated evaluation yet. Run a review to generate metrics.")

    st.subheader("Metrics Trend")
    if st.session_state.judge_history:
        st.line_chart(st.session_state.judge_history, width="stretch")
        st.dataframe(st.session_state.judge_history, width="stretch")
    elif st.session_state.evaluation_history:
        st.line_chart(
            {
                "overall": [row["overall"] for row in st.session_state.evaluation_history],
                "quality_group": [row["quality_group"] for row in st.session_state.evaluation_history],
                "security_group": [row["security_group"] for row in st.session_state.evaluation_history],
                "ai_review_group": [row["ai_review_group"] for row in st.session_state.evaluation_history],
                "pii_exposure": [row["pii_exposure"] for row in st.session_state.evaluation_history],
                "hallucination_risk": [row["hallucination_risk"] for row in st.session_state.evaluation_history],
                "quality": [row["quality"] for row in st.session_state.evaluation_history],
                "security": [row["security"] for row in st.session_state.evaluation_history],
                "maintainability": [
                    row["maintainability"] for row in st.session_state.evaluation_history
                ],
                "reliability": [row["reliability"] for row in st.session_state.evaluation_history],
            }
        )
        st.dataframe(st.session_state.evaluation_history, width="stretch")
    else:
        st.info("No evaluation trend data yet.")

with tab_tracing:
    st.subheader("LangSmith Tracing")
    st.write(f"Tracing enabled: {'Yes' if enable_langsmith else 'No'}")
    st.write(f"Project: {langsmith_project if enable_langsmith else 'N/A'}")
    st.write(f"Endpoint: {langsmith_endpoint if enable_langsmith else 'N/A'}")

    if st.button("Check LangSmith Connection", width="stretch"):
        if not enable_langsmith:
            st.session_state.langsmith_status = {
                "ok": False,
                "message": "Tracing is disabled.",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        elif not active_langsmith_key:
            st.session_state.langsmith_status = {
                "ok": False,
                "message": "LangSmith API key is missing.",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            try:
                client = Client(api_key=active_langsmith_key, api_url=langsmith_endpoint)
                info = client.info()
                st.session_state.langsmith_status = {
                    "ok": True,
                    "message": f"Connected. Instance: {getattr(info, 'tenant_handle', 'unknown')}",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            except Exception as exc:
                st.session_state.langsmith_status = {
                    "ok": False,
                    "message": str(exc),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

    if st.session_state.langsmith_status:
        status = st.session_state.langsmith_status
        if status["ok"]:
            st.success(f"{status['time']}: {status['message']}")
        else:
            st.error(f"{status['time']}: {status['message']}")

    if st.session_state.last_review_run_url or st.session_state.last_fix_run_url:
        st.caption("Latest trace URLs")
        if st.session_state.last_review_run_url:
            st.write(f"Review trace: {st.session_state.last_review_run_url}")
        if st.session_state.last_fix_run_url:
            st.write(f"Fix trace: {st.session_state.last_fix_run_url}")

    st.subheader("LangSmith Evaluation")
    eval_col1, eval_col2, eval_col3 = st.columns([1, 1, 2])
    with eval_col1:
        review_score = st.slider("Review quality", 1, 5, 4)
    with eval_col2:
        fix_score = st.slider("Fix quality", 1, 5, 4)
    with eval_col3:
        eval_comment = st.text_input("Evaluation note", value="")

    if st.button("Submit Evaluation", width="stretch"):
        if not enable_langsmith:
            st.error("Enable LangSmith tracing first.")
        elif not active_langsmith_key:
            st.error("Missing LangSmith API key.")
        elif not st.session_state.last_review_run_id and not st.session_state.last_fix_run_id:
            st.error("No traced run found. Run review first.")
        else:
            try:
                client = Client(api_key=active_langsmith_key, api_url=langsmith_endpoint)
                if st.session_state.last_review_run_id:
                    submit_langsmith_feedback(
                        client,
                        st.session_state.last_review_run_id,
                        review_score,
                        "review_quality",
                        eval_comment,
                    )
                if st.session_state.last_fix_run_id:
                    submit_langsmith_feedback(
                        client,
                        st.session_state.last_fix_run_id,
                        fix_score,
                        "fix_quality",
                        eval_comment,
                    )

                st.session_state.evaluation_logs.append(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "review_score": review_score,
                        "fix_score": fix_score,
                        "comment": eval_comment,
                        "review_run_id": st.session_state.last_review_run_id,
                        "fix_run_id": st.session_state.last_fix_run_id,
                    }
                )
                st.success("Evaluation submitted to LangSmith.")
            except Exception as exc:
                st.error(f"Failed to submit evaluation: {exc}")

    st.subheader("Evaluation Logs")
    if st.session_state.evaluation_logs:
        st.dataframe(st.session_state.evaluation_logs, width="stretch")
    else:
        st.info("No evaluation logs yet.")

with tab_flow:
    st.subheader("Agentic Flow")
    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            node [shape=box, style=rounded];
            input [label="Code Input\n(Paste/Upload)"];
            review [label="Agent 1\nCode Review + static_code_check"];
            findings [label="Review Findings"];
            fix [label="Agent 2\nFix Suggestions"];
            outputs [label="UI Outputs\nReview + Fixes"];
            tracing [label="LangSmith Tracing\n(optional)"];
            monitoring [label="Monitoring\nTokens + Activity Logs"];
            input -> review -> findings -> fix -> outputs;
            review -> tracing;
            fix -> tracing;
            outputs -> monitoring;
        }
        """
    )

    latest = st.session_state.latest_internals
    if latest:
        top1, top2, top3 = st.columns(3)
        top1.metric("Last Run", latest["run"])
        top2.metric("Duration (sec)", latest["duration_sec"])
        top3.metric("Input Lines", latest["input_lines"])

        with st.expander("Prompts Used", expanded=False):
            st.write("Reviewer Prompt")
            st.code(latest["review_prompt"], language="text")
            st.write("Fixer Prompt")
            st.code(latest["fixer_prompt"], language="text")

        with st.expander("Token Breakdown", expanded=True):
            st.json(
                {
                    "review_usage": latest["review_usage"],
                    "fix_usage": latest["fix_usage"],
                    "total_usage": latest["total_usage"],
                }
            )

        with st.expander("Message Internals", expanded=False):
            st.write("Reviewer messages")
            st.dataframe(latest["review_messages"], width="stretch")
            st.write("Fixer messages")
            st.dataframe(latest["fix_messages"], width="stretch")

        with st.expander("Trace Metadata", expanded=False):
            st.json(
                {
                    "review_trace": latest["review_trace"],
                    "fix_trace": latest["fix_trace"],
                    "model": latest["model"],
                    "language": latest["language"],
                    "source": latest["source"],
                    "timestamp": latest["timestamp"],
                }
            )
    else:
        st.info("Run at least one review to inspect internal flow details.")
