import os
import time
from datetime import datetime
from typing import Any

import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.tracers.langchain import LangChainTracer
from langchain_openai import ChatOpenAI
from langsmith import Client


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
            "review_output",
            "fix_output",
            "last_review_run_id",
            "last_fix_run_id",
            "last_review_run_url",
            "last_fix_run_url",
            "langsmith_status",
            "latest_internals",
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

tab_review, tab_monitoring, tab_tracing, tab_flow = st.tabs(
    ["Review", "Monitoring", "Tracing", "Agentic Flow"]
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
