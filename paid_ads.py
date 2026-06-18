from shared import AgentState, banner, WHITE, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

PAID_ADS_SYSTEM = """You are a Paid Advertising Agent.

Design advertising campaigns for:

  • Google Ads
  • Meta Ads
  • LinkedIn Ads
  • YouTube Ads

Provide:
  • Campaign structure
  • Audience targeting
  • Budget recommendations
  • Ad copy
  • Creative ideas
  • Optimization suggestions

Focus on ROI and lead generation.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous Paid Advertising Reviewer.
You will be given the original campaign brief, budget constraints, or product source materials,
and the paid advertising campaign that was generated from them.
Review the campaign for:
  • Strategic alignment — flag any campaign elements (targeting, budgets, copy) that are not supported by, or
    contradict, the supplied source materials or cross-platform objectives
  • Completeness against the required sections (Campaign structure, Audience targeting,
    Budget recommendations, Ad copy, Creative ideas, Optimization suggestions) across the
    relevant platforms (Google, Meta, LinkedIn, YouTube)
  • Optimization for ROI and lead generation — ensure the strategies allocate resources efficiently
    and follow platform-specific best practices to maximize returns
  • Clarity, logical structure, and correct Markdown formatting
  • Platform-inappropriate ad copy, unrealistic budget distributions, or generic creative ideas
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the campaign is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Paid Advertising Agent revising your own report based on
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



@traceable(name="paid_ads")
def PAID_ADS_node(state: AgentState) -> dict:
    PAID_ADS_search_query = state["task"]
    PAID_ADS_search_query+= " for Paid Ads perspective"
    PAID_ADS_raw_results = ddg_search(query=PAID_ADS_search_query, max_results=8)   
    banner("PAID ADS", f"Creates paid advertising campaigns for → {state['task'][:200]}.", WHITE)
    response = llm.invoke([
        SystemMessage(content=PAID_ADS_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Paid ADs search results:\n{PAID_ADS_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("PAID ADS", f"Sending draft to reviewer (round {review_rounds})", WHITE)

        review = _review_report(state["task"], output, PAID_ADS_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("PAID ADS", "Reviewer approved the draft – no further changes", WHITE)
            break

        banner("PAID ADS", f"Reviewer raised {len(comments)} comment(s) – revising", WHITE)
        output = _revise_report(state["task"], output, comments, PAID_ADS_raw_results)
    else:
        banner("PAID ADS", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", WHITE)

    title  = "Paid Ads Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("PAID ADS", f"Paid Ads completed its task – invoking write_document tool", WHITE)
    paid_ads_decision = "PAID_ADS_COMPLETED"

    banner("PAID ADS", f"Decision → {paid_ads_decision.upper()}", WHITE)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Paid Ads Analysis:  → {summary}",f"[Paid Ads] : route → {paid_ads_decision}"]
    
    return {
        "paid_ads_decision": paid_ads_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }