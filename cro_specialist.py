from shared import AgentState, banner, GRAY, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from meta_ads import META_ADS_SYSTEM

CRO_SPECIALIST_SYSTEM = """You are a Conversion Rate Optimization Agent.

Analyze conversion funnels and recommend improvements.

Provide:
  • Funnel analysis
  • UX recommendations
  • CTA improvements
  • A/B test ideas
  • Conversion bottlenecks
  • Landing page suggestions

Focus on measurable conversion improvements.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous CRO Reviewer.

You will be given the original source data (e.g. funnel/analytics data, page
data, or research) that was used as input, and the CRO report that was written
from it.

Review the report for:
  • Completeness against the required sections (Funnel analysis, UX
    recommendations, CTA improvements, A/B test ideas, Conversion bottlenecks,
    Landing page suggestions)
  • Factual accuracy — flag any claim, number, or bottleneck cited in the report
    that is not supported by, or contradicts, the supplied source data
  • Diagnosis-to-recommendation fit — flag any recommendation that doesn't
    clearly trace back to an identified bottleneck or data point (i.e. fixes
    proposed without a stated problem)
  • Measurability — flag recommendations or test ideas that lack a clear success
    metric or are too vague to evaluate as "improved" or "not improved"
  • A/B test validity — flag test ideas that are poorly formed (no clear
    hypothesis, no single variable isolated, or testing something with
    obviously insufficient traffic/impact to matter)
  • Prioritization — flag a list of recommendations with no indication of which
    bottlenecks matter most or which fixes are highest-impact/lowest-effort
  • Redundant, generic, or unsupported claims
  • Markdown formatting correctness

Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}

If the report is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Conversion Rate Optimization Agent revising your own report based on
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


@traceable(name="cro_specialist")
def CRO_SPECIALIST_node(state: AgentState) -> dict:
    cro_specialist_search_query = state["task"]
    cro_specialist_search_query+= " for CRO Specialist perspective"
    cro_specialist_raw_results = ddg_search(query=cro_specialist_search_query, max_results=8)   
    banner("CRO SPECIALIST", f"Analyzes conversion funnels for → {state['task'][:200]}.", GRAY)
    response = llm.invoke([
        SystemMessage(content=CRO_SPECIALIST_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"CRO Specialist search results:\n{cro_specialist_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("CRO SPECIALIST", f"Sending draft to reviewer (round {review_rounds})", GRAY)

        review = _review_report(state["task"], output, cro_specialist_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("CRO SPECIALIST", "Reviewer approved the draft – no further changes", GRAY)
            break

        banner("CRO SPECIALIST", f"Reviewer raised {len(comments)} comment(s) – revising", GRAY)
        output = _revise_report(state["task"], output, comments, cro_specialist_raw_results)
    else:
        banner("CRO SPECIALIST", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", GRAY)

    title  = "CRO Specialist Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("CRO SPECIALIST", f"CRO Specialist completed its task – invoking write_document tool", GRAY)
    cro_decision = "CRO_COMPLETED"

    banner("CRO SPECIALIST", f"Decision → {cro_decision.upper()}", GRAY)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" CRO Analysis:  → {summary}",f"[CRO Specialist] : route → {cro_decision}"]

    return {
        "cro_decision":   cro_decision,
        "output_file": [tool_result["file"]],
        "messages":       msgs,
    }