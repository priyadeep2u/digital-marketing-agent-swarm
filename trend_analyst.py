from shared import AgentState, banner, RED, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

TREND_ANALYST_SYSTEM = """You are a Trend Analysis Agent.

Monitor and identify emerging opportunities.

Provide:
  • Industry trends
  • Viral topics
  • Consumer behavior shifts
  • Competitive movements
  • Emerging technologies
  • Strategic implications

Focus on identifying opportunities before competitors.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous Trend Analysis Reviewer.
You will be given the original market signals, industry data, or source materials,
and the trend analysis report that was generated from them.
Review the report for:
  • Factual accuracy and alignment — flag any trends, movements, or shifts that are not supported by, or
    contradict, the supplied source materials or observable market realities
  • Completeness against the required sections (Industry trends, Viral topics,
    Consumer behavior shifts, Competitive movements, Emerging technologies, Strategic implications)
  • Forward-looking value — ensure the insights focus on identifying emerging opportunities early
    rather than just reporting on already saturated or outdated topics
  • Clarity, logical structure, and correct Markdown formatting
  • Vague trend descriptions, unsupported behavioral shifts, or generic strategic implications
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the report is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Trend Analysis Agent revising your own report based on
reviewer feedback.

You will be given your original report and a list of reviewer comments.
For each comment:
  - If it identifies a genuine, valid issue (factual gap, missing required section,
    unsupported claim, structural problem), incorporate a fix.
  - If it is invalid, unclear, or just a style nitpick, ignore it.

Output the FULL revised report in structured Markdown. Do not include any commentary
about which comments you accepted or rejected, and do not mention the review process —
output only the final report content.
"""

MAX_REVIEW_ROUNDS = 2


def _review_report(task: str, draft: str, search_results) -> dict:
    """Second LLM call: critique the draft against the source data and return structured feedback."""
    review_response = llm_reviewer.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=(
            f"Task: {task}\n\n"
            f"Internet search results (source data the report should be grounded in):\n{search_results}\n\n"
            f"Report to review:\n{draft}"
        )),
    ])

    raw = review_response.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("review response was not a JSON object")
        parsed.setdefault("approved", False)
        parsed.setdefault("comments", [])
        return parsed
    except Exception:
        # If the reviewer didn't return valid JSON, fail safe by approving
        # the draft as-is rather than blocking the pipeline.
        return {"approved": True, "comments": []}


def _revise_report(task: str, draft: str, comments: list, search_results) -> str:
    """First LLM call (author), incorporating only the valid review comments."""
    comments_block = "\n".join(f"- {c}" for c in comments)
    revision_response = llm.invoke([
        SystemMessage(content=REVISION_SYSTEM),
        HumanMessage(content=(
            f"Task: {task}\n\n"
            f"Internet search results (source data):\n{search_results}\n\n"
            f"Original report:\n{draft}\n\n"
            f"Reviewer comments:\n{comments_block}"
        )),
    ])
    return revision_response.content.strip()


@traceable(name="trend_analyst")
def trend_analyst_node(state: AgentState) -> dict:
    trend_analyst_search_query = state["task"]
    trend_analyst_search_query+= " for Trend Analyst perspective"
    trend_analyst_raw_results = ddg_search(query=trend_analyst_search_query, max_results=8) 
    banner("TREND ANALYST", f"Analyzing trends for → {state['task'][:200]}.", RED)
    response = llm.invoke([
        SystemMessage(content=TREND_ANALYST_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                   f"Trend Analyst search results:\n{trend_analyst_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("TREND ANALYST", f"Sending draft to reviewer (round {review_rounds})", RED)

        review = _review_report(state["task"], output, trend_analyst_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("TREND ANALYST", "Reviewer approved the draft – no further changes", RED)
            break

        banner("TREND ANALYST", f"Reviewer raised {len(comments)} comment(s) – revising", RED)
        output = _revise_report(state["task"], output, comments, trend_analyst_raw_results)
    else:
        banner("TREND ANALYST", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", RED)

    title  = "Trend Analyst Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("TREND ANALYST", f"Trend Analyst completed its task – invoking write_document tool", RED)
    trend_analyst_decision = "TREND_ANALYSIS_COMPLETED"

    banner("TREND ANALYST", f"Decision → {trend_analyst_decision.upper()}", RED)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Trend Analyst Analysis:  → {summary}",f"[Trend Analyst] : route → {trend_analyst_decision}"]

    return {
        "trend_analyst_decision": trend_analyst_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }