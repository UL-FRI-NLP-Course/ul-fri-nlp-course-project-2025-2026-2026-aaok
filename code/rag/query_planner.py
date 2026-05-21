import json
import re

class QueryPlanner:
    def __init__(self, llm, data_cutoff="2024-12-31"):
        self.llm = llm
        self.data_cutoff = data_cutoff

    def build_prompt(self, query: str, detected_start: str, detected_end: str):
        return f"""
You are a legal retrieval query planner for a RAG system.

DATA CONTEXT:
- Dataset contains legal documents up to: {self.data_cutoff}
- Some laws may have validity beyond this date.

YOU ARE GIVEN A PRE-EXTRACTED TIME WINDOW:
- detected_start_date: {detected_start}
- detected_end_date: {detected_end}

Your job:
You may:
1. use this time window as-is
2. modify it
3. ignore it entirely if irrelevant

RULES:
- If query is general legal question → usually ignore time filtering
- If query is about "current rights / today / now" → strongly apply time filter
- If query is historical → use detected range
- If detected year > 2024 → clamp to {self.data_cutoff}

OUTPUT ONLY VALID JSON:

{{
  "rewritten_query": string,
  "intent_keywords": [string],
  "query_specificity_level": "GENERAL" | "SPECIFIC",

  "use_doc_faiss": true,
  "use_chunk_faiss": true,
  "use_bm25": true,

  "apply_time_filter": true | false,
  "time_window": ["YYYY-MM-DD", "YYYY-MM-DD"] | null,

  "reasoning_brief": string
}}

USER QUERY:
{query}
"""

    def parse(self, text: str):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        return json.loads(match.group(0))

    def __call__(self, query: str, detected_start: str, detected_end: str):
        prompt = self.build_prompt(query, detected_start, detected_end)
        response = self.llm.invoke(prompt)
        return self.parse(response.content)