"""
LangGraph agent graph for AcmeCloud KB Q&A.

Flow per conversation turn (happy path):
    retrieve  →  answer  →  save

When evidence is weak (no citations / "cannot answer") and retries remain:
    answer  →  refine_and_retrieve  →  answer  →  save

The conditional edge after `answer` implements the agentic second-pass:
it detects weak evidence and loops back with a refined query before
giving up and saving.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from langgraph.graph import END, StateGraph

from agent.memory import read_memory, write_memory
from agent.output import save_outputs
from agent.retriever import KeywordSearchRetriever, SearchResult
from agent.schema import Answer

# ---------------------------------------------------------------------------
# How many top retriever results to pass to the LLM
# ---------------------------------------------------------------------------
_TOP_K = 5

# How many times the agent may retry with a refined query when evidence is weak
_MAX_RETRIES = 1

# Phrases in the LLM's answer that signal weak / absent evidence
_WEAK_EVIDENCE_PHRASES = (
    "cannot answer",
    "can't answer",
    "don't have information",
    "do not have information",
    "no information",
    "not in the kb",
    "not covered",
    "insufficient",
)

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    turn: int
    query: str                        # current search query (may be refined)
    original_query: str               # user's original question, never mutated
    kb_dir: str
    out_dir: str
    retrieval_results: list[SearchResult]
    answer: Answer | None
    memory: dict[str, Any]
    retry_count: int                  # how many second-pass retries have run


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
6. When memory context is provided, use it directly — never re-ask for information already known.

Memory update rules — save ALL of the following whenever they are known or computed:
- Facts the user states: seat_count, monthly_mau, billing_cycle
- Results you compute: selected_plan, monthly_base, extra_seats_cost, mau_overage_cost, monthly_total
- When answering a follow-up, also persist annual_total and annual_monthly_equivalent if computed.
Use short lowercase_underscore keys. Values must be strings or numbers (no units in the value).

Respond with a single valid JSON object — no markdown fences, no extra text:
{
  "final_answer": "<your answer as plain text>",
  "citations": [{"file": "<kb file path>", "lines": "<start-end>"}],
  "assumptions": ["<any assumption you made>"],
  "memory_updates": [{"key": "<key>", "value": "<value>"}],
  "tool_calls": [{"tool": "search_kb", "input": "<the query that was searched>"}]
}
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

    # Merge LLM-extracted facts into the in-memory state dict (no disk I/O here)
    updated_memory = {**state["memory"], **{mu.key: mu.value for mu in structured.memory_updates}}

    return {**state, "answer": structured, "memory": updated_memory}


# ---------------------------------------------------------------------------
# Conditional routing: decide whether a second-pass retry is warranted
# ---------------------------------------------------------------------------

def _is_weak_evidence(ans: Answer) -> bool:
    """Return True when the LLM signalled it lacked sufficient KB evidence."""
    if not ans.citations:
        return True
    lowered = ans.final_answer.lower()
    return any(phrase in lowered for phrase in _WEAK_EVIDENCE_PHRASES)


def _route_after_answer(state: AgentState) -> str:
    """
    Conditional edge function called after the `answer` node.

    Routes to:
      - "refine_and_retrieve"  when evidence is weak AND retries remain
      - "save"                 otherwise
    """
    ans = state["answer"]
    if ans is not None and _is_weak_evidence(ans) and state["retry_count"] < _MAX_RETRIES:
        print(
            f"  [retry {state['retry_count'] + 1}/{_MAX_RETRIES}] "
            f"Weak evidence detected — refining query…"
        )
        return "refine_and_retrieve"
    return "save"


# ---------------------------------------------------------------------------
# Node: refine_and_retrieve  (second-pass agentic behavior)
# ---------------------------------------------------------------------------

_REFINE_PROMPT = """You are a search query optimizer for a support KB.

The original user question yielded weak or no results.
Your job: rewrite the search query using different keywords, synonyms, or
a more specific sub-question likely to match KB content.

Original question: {original_query}
Current (failed) query: {current_query}
LLM answer so far: {current_answer}

Reply with ONLY the refined search query — no explanation, no quotes."""


def refine_and_retrieve(state: AgentState) -> AgentState:
    """
    Second-pass node: ask the LLM for a better search query, then re-retrieve.

    This implements the agentic self-check loop — when the first answer lacked
    citations or admitted ignorance, we reformulate rather than give up.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    refine_prompt = _REFINE_PROMPT.format(
        original_query=state["original_query"],
        current_query=state["query"],
        current_answer=state["answer"].final_answer if state["answer"] else "(none)",
    )

    refined_query = llm.invoke([{"role": "user", "content": refine_prompt}]).content.strip()
    print(f"  [refine] new query: {refined_query!r}")

    # Re-run retrieval with the refined query
    retriever = KeywordSearchRetriever(state["kb_dir"])
    results = retriever.search(refined_query)

    return {
        **state,
        "query": refined_query,
        "retrieval_results": results[:_TOP_K],
        "retry_count": state["retry_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Node: save
# ---------------------------------------------------------------------------

def save(state: AgentState) -> AgentState:
    """Append this turn's outputs and flush the current memory state to disk."""
    save_outputs(state["out_dir"], state["turn"], state["query"], state["answer"])
    write_memory(state["out_dir"], state["memory"])
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("refine_and_retrieve", refine_and_retrieve)
    g.add_node("save", save)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")

    # Conditional second-pass: retry with refined query when evidence is weak
    g.add_conditional_edges(
        "answer",
        _route_after_answer,
        {"refine_and_retrieve": "refine_and_retrieve", "save": "save"},
    )
    g.add_edge("refine_and_retrieve", "answer")  # loop back for re-answering
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
            "turn": turn["turn"],
            "query": query,
            "original_query": query,
            "kb_dir": kb_dir,
            "out_dir": out_dir,
            "retrieval_results": [],
            "answer": None,
            "memory": memory,
            "retry_count": 0,
        }

        final_state = graph.invoke(state)
        memory = final_state["memory"]
        ans = final_state["answer"]

        print(f"  → {ans.final_answer[:120]}")
        for c in ans.citations:
            print(f"     cite: {c.file}:{c.lines}")
