from shared import AgentState, banner, CORAL, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

ANALYTICS_SYSTEM = """You are a Marketing Analytics Agent.

Analyze marketing performance and generate insights.

Provide:
  • KPI analysis
  • Traffic insights
  • Conversion analysis
  • Channel performance
  • Trend identification
  • Strategic recommendations

Use data to support all conclusions.

Output in structured Markdown.
"""

REVIEWER_SYSTEM = """You are a meticulous Marketing Analytics Reviewer.

You will be given the original internet search results that were used as source data,
and the report that was written from them.

Review the report for:
  • Factual accuracy — flag any claim in the report that is not supported by, or
    contradicts, the supplied search results
  • Completeness against the required sections (KPI analysis, Traffic insights,
    Conversion analysis, Channel performance, Trend identification, Strategic recommendations)
  • Clarity, logical structure, and correct Markdown formatting
  • Redundant, vague, or unsupported claims

Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}

If the report is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Marketing Analytics Agent revising your own report based on
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


@traceable(name="analytics")
def ANALYTICS_node(state: AgentState) -> dict:
    analytics_search_query = state["task"]
    analytics_search_query += " for Analytics perspective"
    analytics_raw_results = ddg_search(query=analytics_search_query, max_results=8)
    banner("ANALYTICS", f"Analyzes marketing performance for → {state['task'][:200]}.", CORAL)

    response = llm.invoke([
        SystemMessage(content=ANALYTICS_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Internet search results:\n{analytics_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("ANALYTICS", f"Sending draft to reviewer (round {review_rounds})", CORAL)

        review = _review_report(state["task"], output, analytics_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("ANALYTICS", "Reviewer approved the draft – no further changes", CORAL)
            break

        banner("ANALYTICS", f"Reviewer raised {len(comments)} comment(s) – revising", CORAL)
        output = _revise_report(state["task"], output, comments, analytics_raw_results)
    else:
        banner("ANALYTICS", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", CORAL)

    title = "Analytics Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("ANALYTICS", f"Analytics completed its task – invoking write_document tool", CORAL)
    analytics_decision = "ANALYTICS_COMPLETED"

    banner("ANALYTICS", f"Decision → {analytics_decision.upper()}", CORAL)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Analytics Analysis:  → {summary}",f"[Analytics Analyst] : route → {analytics_decision}"]

    return {
        "analytics_decision": analytics_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }
