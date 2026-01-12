from typing import TypedDict, Optional

class AgentState(TypedDict):
    user_query: str
    raw_tool_output: str
    analyst_notes: Optional[str]
    verified_output: Optional[str]
    final_answer: Optional[str]
