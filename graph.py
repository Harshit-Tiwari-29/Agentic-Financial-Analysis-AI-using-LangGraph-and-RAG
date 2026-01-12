from langgraph.graph import StateGraph, END
from graph_state import AgentState
from agents import analyst_agent, verifier_agent, summarizer_agent

def build_postprocessing_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyst", analyst_agent)
    graph.add_node("verifier", verifier_agent)
    graph.add_node("summarizer", summarizer_agent)

    graph.set_entry_point("analyst")
    graph.add_edge("analyst", "verifier")
    graph.add_edge("verifier", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()
