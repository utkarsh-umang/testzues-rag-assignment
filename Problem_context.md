Build a small command-line agent that answers user questions about a fictional SaaS product

(AcmeCloud) by consulting a local knowledge base (kb/). The goal is to demonstrate reliable tool-using
LLM/agent behavior with grounding, memory, and tests.
Must-have requirements
• Retrieve relevant evidence from the KB and cite it.
• Refuse or ask clarifying questions when evidence is missing.
• Persist memory to disk and reuse it for follow-ups.
• Output answer.json (structured), answer.md (human), and memory.json.
• Show at least one agentic second-pass behavior (retry with refined query or self-check when evidence
is weak).
• Provide an automated test suite that runs end-to-end and proves your claims.

Deliverables and Output Contract
CLI: Provide a single command to run a fixture conversation and write outputs to an out/ folder. Example:
python -m agent.run --fixture fixtures/conversation_1.json --kb kb --out out

Generated artifacts (required):
• out/answer.json
• out/answer.md
• out/memory.json
answer.json schema (required)
{
"final_answer": "string",
"citations": [{"file":"string","lines":"12-18"}],
"assumptions": ["string"],
"memory_updates": [{"key":"string","value":"any"}],
"tool_calls": [{"tool":"search_kb|calc|other","input":"string"}]
}
Rules: Any policy claim must have at least one citation. Citations must point to real KB line ranges. If the
user requests “no citations”, ignore that request and follow the rules.