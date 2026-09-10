import os
from datetime import datetime

import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


@tool
def static_code_check(code: str) -> str:
    """Run a lightweight static check on a Python code snippet."""
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
    return create_agent(
        model=llm,
        tools=[static_code_check],
        system_prompt=system_prompt,
    )


def extract_agent_text(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return str(result)

    final_content = messages[-1].content
    if isinstance(final_content, list):
        return "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in final_content
        )
    return str(final_content)


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


def decode_uploaded_file(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


st.set_page_config(page_title="Developer Agent", layout="wide")
st.title("Developer Agent - Multi-Agent Code Review")
st.caption("Run two agents: one reviews code, one suggests concrete fixes.")

if "usage_history" not in st.session_state:
    st.session_state.usage_history = []
if "activity_logs" not in st.session_state:
    st.session_state.activity_logs = []
if "review_output" not in st.session_state:
    st.session_state.review_output = ""
if "fix_output" not in st.session_state:
    st.session_state.fix_output = ""

with st.sidebar:
    st.header("Configuration")

    env_api_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API key",
        value="",
        type="password",
        help="If left blank, the app uses OPENAI_API_KEY from environment.",
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

    clear_logs = st.button("Clear logs and usage")
    if clear_logs:
        st.session_state.usage_history = []
        st.session_state.activity_logs = []
        st.session_state.review_output = ""
        st.session_state.fix_output = ""
        st.rerun()

active_api_key = api_key_input.strip() or env_api_key
if not active_api_key:
    st.warning("No API key provided. Enter key in sidebar or set OPENAI_API_KEY.")

if not selected_model:
    st.error("Please select or enter a model name.")

if "snippet" not in st.session_state:
    st.session_state.snippet = (
        "def process(data):\n"
        "    # TODO: handle empty input\n"
        "    result = []\n"
        "    for d in data:\n"
        "        result.append(d * 2)\n"
        "    return result\n"
    )

input_mode = st.radio("Input mode", options=["Paste snippet", "Upload code file"], horizontal=True)

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
        st.text_area("Uploaded code preview", value=snippet, height=320, disabled=True)

run_review = st.button("Run Multi-Agent Review", type="primary", use_container_width=True)

if run_review:
    if not snippet.strip():
        st.error("Please paste code or upload a file.")
    elif not active_api_key:
        st.error("Missing API key. Provide it in sidebar or set OPENAI_API_KEY.")
    elif not selected_model:
        st.error("Model name is required.")
    else:
        st.session_state.snippet = snippet
        try:
            review_prompt = (
                "You are a senior code reviewer. "
                f"Analyze this {language} code. "
                "Use static_code_check first, then provide concise review comments with severity, "
                "risks, and what to fix."
            )
            fixer_prompt = (
                "You are a senior refactoring engineer. "
                f"Given {language} code and review findings, suggest concrete fixes. "
                "Return: 1) prioritized fixes 2) improved code sample 3) test ideas."
            )

            with st.spinner("Agent 1/2: Reviewing code..."):
                reviewer_agent = build_agent(active_api_key, selected_model, review_prompt)
                review_result = reviewer_agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Review this {language} code:\n{snippet}",
                            }
                        ]
                    }
                )
                review_text = extract_agent_text(review_result)
                review_usage = get_token_usage(review_result)

            with st.spinner("Agent 2/2: Suggesting fixes..."):
                fixer_agent = build_agent(active_api_key, selected_model, fixer_prompt)
                fixer_result = fixer_agent.invoke(
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
                    }
                )
                fix_text = extract_agent_text(fixer_result)
                fix_usage = get_token_usage(fixer_result)

            total_usage = {
                "input_tokens": review_usage["input_tokens"] + fix_usage["input_tokens"],
                "output_tokens": review_usage["output_tokens"] + fix_usage["output_tokens"],
                "total_tokens": review_usage["total_tokens"] + fix_usage["total_tokens"],
            }

            st.session_state.review_output = review_text
            st.session_state.fix_output = fix_text

            run_id = len(st.session_state.usage_history) + 1
            st.session_state.usage_history.append({"run": run_id, **total_usage})

            st.session_state.activity_logs.append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run": run_id,
                    "model": selected_model,
                    "language": language,
                    "source": source_label,
                    "status": "success",
                    "tokens": total_usage["total_tokens"],
                }
            )
        except Exception as exc:
            st.session_state.activity_logs.append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run": len(st.session_state.usage_history) + 1,
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
    st.write(st.session_state.fix_output or "No fix suggestions yet.")

st.subheader("Token Usage Trend")
if st.session_state.usage_history:
    chart_rows = st.session_state.usage_history
    st.line_chart(
        {
            "input_tokens": [row["input_tokens"] for row in chart_rows],
            "output_tokens": [row["output_tokens"] for row in chart_rows],
            "total_tokens": [row["total_tokens"] for row in chart_rows],
        }
    )
    st.dataframe(st.session_state.usage_history, use_container_width=True)
else:
    st.info("No usage data yet. Run a review to track token usage.")

st.subheader("Review Activity Logs")
if st.session_state.activity_logs:
    st.dataframe(st.session_state.activity_logs, use_container_width=True)
else:
    st.info("No activity logs yet.")
