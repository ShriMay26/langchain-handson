import os

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


def build_agent(api_key: str):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    return create_agent(
        model=llm,
        tools=[static_code_check],
        system_prompt=(
            "You are a senior developer performing a code review. "
            "Use static_code_check on the snippet, then write a short, "
            "constructive review comment covering what to fix and why."
        ),
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


st.set_page_config(page_title="Developer Agent", page_icon="🧑‍💻", layout="wide")
st.title("Developer Agent - Code Reviewer")
st.caption("Paste Python code and get a concise review powered by LangChain + OpenAI.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("OPENAI_API_KEY is not set. Set it in your terminal before running Streamlit.")

if "snippet" not in st.session_state:
    st.session_state.snippet = (
        "def process(data):\n"
        "    # TODO: handle empty input\n"
        "    result = []\n"
        "    for d in data:\n"
        "        result.append(d * 2)\n"
        "    return result\n"
    )

snippet = st.text_area(
    "Python snippet",
    value=st.session_state.snippet,
    height=320,
    key="snippet_editor",
)

if st.button("Review code", type="primary", use_container_width=True):
    if not snippet.strip():
        st.error("Please enter code to review.")
    elif not api_key:
        st.error("Missing OPENAI_API_KEY. Configure it and retry.")
    else:
        with st.spinner("Reviewing code..."):
            try:
                agent = build_agent(api_key)
                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Review this code:\n{snippet}",
                            }
                        ]
                    }
                )
                review_text = extract_agent_text(result)
                st.subheader("Review output")
                st.write(review_text)
            except Exception as exc:
                st.error(f"Review failed: {exc}")
