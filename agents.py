import os
from langchain_groq import ChatGroq


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment")

    return ChatGroq(
        api_key=api_key,
        model="llama3-8b-8192",
        temperature=0
    )


# llm = ChatGroq(model="llama3-8b-8192", temperature=0)

def analyst_agent(state: dict):
    prompt = f"""
Explain what the system did to answer the query.
Do NOT add new information.

Output:
{state["raw_tool_output"]}
"""
    state["analyst_notes"] = llm.invoke(prompt).content
    return state

def verifier_agent(state: dict):
    prompt = f"""
Verify the following output.
Check only for hallucinations or obvious errors.
Do NOT add new facts.

Output:
{state["raw_tool_output"]}
"""
    state["verified_output"] = llm.invoke(prompt).content
    return state

def summarizer_agent(state: dict):
    prompt = f"""
Convert this into a concise, user-facing answer:

{state["verified_output"]}
"""
    state["final_answer"] = llm.invoke(prompt).content
    return state
