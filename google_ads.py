from shared import AgentState, banner, DARK_GREEN, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

GOOGLE_ADS_SYSTEM = """You are a Google Ads Specialist.

Create high-performing search and display campaigns.

Provide:
  • Campaign structure
  • Ad groups
  • Keywords
  • Negative keywords
  • Ad copy
  • Extensions
  • Bid recommendations

Optimize for conversions and ROAS.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous Google Ads Reviewer.
You will be given the original campaign brief, landing page details, or product source materials,
and the Google Ads campaign that was generated from them.
Review the campaign for:
  • Strategic alignment — flag any campaign element (keywords, copy) that is not supported by, or
    contradicts, the supplied source materials or target audience needs
  • Completeness against the required sections (Campaign structure, Ad groups,
    Keywords, Negative keywords, Ad copy, Extensions, Bid recommendations)
  • Optimization for conversions and ROAS — ensure best practices are used (e.g., tight thematic ad groups,
    compelling ad copy with clear CTAs, appropriate negative keywords to prevent wasted spend)
  • Clarity, logical structure, and correct Markdown formatting
  • Keyword cannibalization, policy-violating ad copy, or unrealistic bid recommendations
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the campaign is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Google Ads Specialist revising your own report based on
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


@traceable(name="google_ads")
def GOOGLE_ADS_node(state: AgentState) -> dict:
    google_ads_search_query = state["task"]
    google_ads_search_query+= " for Google ADs perspective"
    google_ads_raw_results = ddg_search(query=google_ads_search_query, max_results=8)   
    banner("GOOGLE ADS", f"Creates Google Ads campaigns for → {state['task'][:200]}.", DARK_GREEN)
    response = llm.invoke([
        SystemMessage(content=GOOGLE_ADS_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Google Ads search results:\n{google_ads_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("GOOGLE ADS", f"Sending draft to reviewer (round {review_rounds})", DARK_GREEN)

        review = _review_report(state["task"], output, google_ads_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("GOOGLE ADS", "Reviewer approved the draft – no further changes", DARK_GREEN)
            break

        banner("GOOGLE ADS", f"Reviewer raised {len(comments)} comment(s) – revising", DARK_GREEN)
        output = _revise_report(state["task"], output, comments, google_ads_raw_results)
    else:
        banner("GOOGLE ADS", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", DARK_GREEN)

    title  = "Google ADs Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("GOOGLE ADS", f"Google Ads completed its task – invoking write_document tool", DARK_GREEN)
    google_ads_decision = "GOOGLE_ADS_COMPLETED"

    banner("GOOGLE ADS", f"Decision → {google_ads_decision.upper()}", DARK_GREEN)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Google Ads Analysis:  → {summary}",f"[Google Ads] : route → {google_ads_decision}"]

    return {
        "google_ads_decision": google_ads_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }