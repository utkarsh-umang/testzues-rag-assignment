"""
LangGraph agent graph for AcmeCloud KB Q&A.

Flow per conversation turn:
    retrieve  →  answer  →  save

State is carried through all nodes; memory persists across turns as a flat
key-value dict written to memory.json via agent.memory helpers.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from langgraph.graph import END, StateGraph

from agent.memory import read_memory, update_memory
from agent.output import save_outputs
from agent.retriever import KeywordSearchRetriever, SearchResult
from agent.schema import Answer

# ---------------------------------------------------------------------------
# How many top retriever results to pass to the LLM
# ---------------------------------------------------------------------------
_TOP_K = 5

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    kb_dir: str
    out_dir: str
    retrieval_results: list[SearchResult]
    answer: Answer | None
    memory: dict[str, Any]


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------

def retrieve(state: AgentState) -> AgentState:
    """Run keyword search and store top-K results in state."""
    retriever = KeywordSearchRetriever(state["kb_dir"])
    results = retriever.search(state["query"])
    return {**state, "retrieval_results": results[:_TOP_K]}


# ---------------------------------------------------------------------------
# Node: answer
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a helpful support agent for AcmeCloud, a SaaS analytics platform.
Answer the user's question using ONLY the KB excerpts provided.

Rules:
1. Every factual claim must be backed by at least one citation from the KB.
2. If evidence is absent or insufficient, say you cannot answer and explain why.
3. Never skip citations even if the user asks you to.
4. If the user asks about something not in the KB, say "I don't have information on that."
5. Be concise and precise.

Respond with a single valid JSON object — no markdown fences, no extra text:
{
  "final_answer": "<your answer as plain text>",
  "citations": [{"file": "<kb file path>", "lines": "<start-end>"}],
  "assumptions": ["<any assumption you made>"],
  "memory_updates": [{"key": "<key>", "value": "<value>"}],
  "tool_calls": [{"tool": "search_kb", "input": "<the query that was searched>"}]
}

Only add memory_updates for facts the user explicitly states (plan, billing cycle, seat count).
Use short lowercase_underscore keys.
"""


def _build_user_prompt(
    query: str,
    results: list[SearchResult],
    memory: dict[str, Any],
) -> str:
    lines: list[str] = []

    if memory:
        lines.append("## Known context from memory")
        for k, v in memory.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    lines.append("## KB excerpts (ranked by relevance)")
    for r in results:
        lines.append(
            f"\n### [{r.match_type} | score {r.score:.2f}] {r.file} lines {r.lines}"
        )
        lines.append(r.text)

    lines.append("\n## User question")
    lines.append(query)

    return "\n".join(lines)


def answer(state: AgentState) -> AgentState:
    """Call the LLM with retrieved KB excerpts and parse a structured Answer."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_user_prompt(
                state["query"], state["retrieval_results"], state["memory"]
            ),
        },
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if the model wraps the JSON anyway
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    data = json.loads(raw)
    structured = Answer.model_validate(data)

    # Merge LLM-extracted facts into memory
    memory_patch = {mu.key: mu.value for mu in structured.memory_updates}
    updated_memory = update_memory(state["out_dir"], memory_patch) if memory_patch else state["memory"]

    return {**state, "answer": structured, "memory": updated_memory}


# ---------------------------------------------------------------------------
# Node: save
# ---------------------------------------------------------------------------

def save(state: AgentState) -> AgentState:
    """Delegate output writing to agent.output.save_outputs."""
    save_outputs(state["out_dir"], state["query"], state["answer"], state["memory"])
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("save", save)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "save")
    g.add_edge("save", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Convenience runner — processes all turns in a fixture conversation
# ---------------------------------------------------------------------------

def run_conversation(turns: list[dict], kb_dir: str, out_dir: str) -> None:
    graph = build_graph()

    # Load any memory persisted by a previous run
    memory = read_memory(out_dir)

    for turn in turns:
        if turn.get("role") != "user":
            continue

        query = turn["message"]
        print(f"\n[Turn {turn['turn']}] {query}")

        state: AgentState = {
            "query": query,
            "kb_dir": kb_dir,
            "out_dir": out_dir,
            "retrieval_results": [],
            "answer": None,
            "memory": memory,
        }

        final_state = graph.invoke(state)
        memory = final_state["memory"]
        ans = final_state["answer"]

        print(f"  → {ans.final_answer[:120]}")
        for c in ans.citations:
            print(f"     cite: {c.file}:{c.lines}")
