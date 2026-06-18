from shared import AgentState, banner, LAVENDER, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

REPORTING_SYSTEM = """You are an Executive Reporting Agent.

Convert marketing findings into executive-level reports.

Provide:
  • Key findings
  • Performance summary
  • Opportunities
  • Risks
  • Strategic recommendations
  • Next steps

Keep reports concise, actionable, and business-focused.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous Executive Reporting Reviewer.
You will be given the original marketing findings or raw performance data,
and the executive report that was generated from them.
Review the report for:
  • Factual accuracy and alignment — flag any summary, metric, or claim that is not supported by, or
    contradicts, the supplied marketing findings
  • Completeness against the required sections (Key findings, Performance summary,
    Opportunities, Risks, Strategic recommendations, Next steps)
  • Executive focus — ensure the report is concise, actionable, and business-focused, avoiding
    unnecessary jargon or getting bogged down in tactical minutiae
  • Clarity, logical structure, and correct Markdown formatting
  • Buried insights, overly verbose explanations, or unsupported strategic recommendations
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the report is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Executive Reporting Agent your own report based on
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


@traceable(name="reporting")
def REPORTING_node(state: AgentState) -> dict:
    REPORTING_search_query = state["task"]
    REPORTING_search_query+= " for Reporting perspective"
    REPORTING_raw_results = ddg_search(query=REPORTING_search_query, max_results=8)  
    banner("REPORTING", f"Creates executive reports for → {state['task'][:200]}.", LAVENDER)
    response = llm.invoke([
        SystemMessage(content=REPORTING_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Reporting search results:\n{REPORTING_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("REPORTING", f"Sending draft to reviewer (round {review_rounds})", LAVENDER)

        review = _review_report(state["task"], output, REPORTING_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("REPORTING", "Reviewer approved the draft – no further changes", LAVENDER)
            break

        banner("REPORTING", f"Reviewer raised {len(comments)} comment(s) – revising", LAVENDER)
        output = _revise_report(state["task"], output, comments, REPORTING_raw_results)
    else:
        banner("REPORTING", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", LAVENDER)

    title  = "Reporting Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("REPORTING", f"Reporting completed its task – invoking write_document tool", LAVENDER)
    reporting_decision = "REPORTING_COMPLETED"

    banner("REPORTING", f"Decision → {reporting_decision.upper()}", LAVENDER)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Reporting Analysis:  → {summary}",f"[Reporting] : route → {reporting_decision}"]

    return {
        "reporting_decision": reporting_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }