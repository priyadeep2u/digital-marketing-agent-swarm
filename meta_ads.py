from shared import AgentState, banner, DARK_YELLOW, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

META_ADS_SYSTEM = """You are a Meta Ads Specialist.

Design Facebook and Instagram advertising campaigns.

Provide:
  • Target audiences
  • Creative concepts
  • Ad copy
  • Funnel structure
  • Budget allocation
  • Optimization recommendations

Focus on engagement and conversion efficiency.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous Meta Ads Reviewer.
You will be given the original campaign brief, target audience details, or product source materials,
and the Meta Ads campaign that was generated from them.
Review the campaign for:
  • Strategic alignment — flag any campaign elements (audiences, creative, copy) that are not supported by, or
    contradict, the supplied source materials or campaign goals
  • Completeness against the required sections (Target audiences, Creative concepts,
    Ad copy, Funnel structure, Budget allocation, Optimization recommendations)
  • Optimization for engagement and conversion efficiency — ensure the funnel makes logical sense
    and that ad copy/creative concepts follow platform best practices
  • Clarity, logical structure, and correct Markdown formatting
  • Policy-violating ad copy, generic creative concepts, or unrealistic budget allocations
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the campaign is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Meta Ads Specialist revising your own report based on
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


@traceable(name="meta_ads")
def META_ADS_node(state: AgentState) -> dict:
    Meta_ADs_search_query = state["task"]
    Meta_ADs_search_query+= " for Meta ADs perspective"
    Meta_ADs_raw_results = ddg_search(query=Meta_ADs_search_query, max_results=8)  
    banner("META ADS", f"Creates Meta Ads campaigns for → {state['task'][:200]}.", DARK_YELLOW)
    response = llm.invoke([
        SystemMessage(content=META_ADS_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Meta ADs search results:\n{Meta_ADs_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("META ADS", f"Sending draft to reviewer (round {review_rounds})", DARK_YELLOW)

        review = _review_report(state["task"], output, Meta_ADs_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("META ADS", "Reviewer approved the draft – no further changes", DARK_YELLOW)
            break

        banner("META ADS", f"Reviewer raised {len(comments)} comment(s) – revising", DARK_YELLOW)
        output = _revise_report(state["task"], output, comments, Meta_ADs_raw_results)
    else:
        banner("META ADS", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", DARK_YELLOW)

    title  = "Meta Ads Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("META ADS", f"Meta Ads completed its task – invoking write_document tool", DARK_YELLOW)
    meta_ads_decision = "META_ADS_COMPLETED"

    banner("META ADS", f"Decision → {meta_ads_decision.upper()}", DARK_YELLOW)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Meta Ads Analysis:  → {summary}",f"[Meta Ads] : route → {meta_ads_decision}"]

    return {
        "meta_ads_decision": meta_ads_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }