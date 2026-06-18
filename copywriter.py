from shared import AgentState, banner, DARK_BLUE, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

COPYWRITER_SYSTEM = """You are a Conversion Copywriting Agent.

Write persuasive copy for:

  • Landing pages
  • Advertisements
  • Emails
  • Product pages
  • Sales funnels

Apply frameworks such as:
  • AIDA
  • PAS
  • BAB

Focus on:
  • Benefits
  • Emotional triggers
  • Clear CTAs
  • Conversion optimization

Output polished copy in Markdown.
"""

REVIEWER_SYSTEM = """You are a meticulous Conversion Copy Reviewer.

You will be given the original brief or context (product/offer details, target
audience, format — landing page, ad, email, product page, or funnel — and any
source material on features/benefits) and the copy that was written from it.

Review the copy for:
  • Claim accuracy — flag any benefit, feature, statistic, or claim that is not
    supported by, or contradicts, the supplied source material, or that overstates
    results in a way that risks misleading the reader
  • Framework execution — confirm the copy coherently follows its intended
    framework (AIDA, PAS, BAB, etc.) with no missing or muddled stages
  • Benefit vs. feature balance — flag copy that lists features without
    translating them into concrete reader benefits
  • CTA strength — flag CTAs that are missing, vague, weak, or misaligned with
    the stated conversion goal
  • Emotional trigger fit — flag emotional appeals that feel manipulative,
    mismatched to the audience, or inconsistent with brand tone
  • Format fit — flag copy that's too long/short or structurally wrong for the
    stated format (e.g. a landing page block written like an email, an ad with
    landing-page-length copy)
  • Redundant, generic, or filler phrasing that dilutes persuasive impact
  • Markdown formatting correctness

Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}

If the copy is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Conversion Copywriting Agent revising your own report based on
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


@traceable(name="copywriter")
def COPYWRITER_node(state: AgentState) -> dict:
    copywriter_search_query = state["task"]
    copywriter_search_query+= " for Copywriter perspective"
    copywriter_raw_results = ddg_search(query=copywriter_search_query, max_results=8)  
    banner("COPYWRITER", f"Creates copy for → {state['task'][:200]}.", DARK_BLUE)
    response = llm.invoke([
        SystemMessage(content=COPYWRITER_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Copywriter search results:\n{copywriter_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("COPYWRITER", f"Sending draft to reviewer (round {review_rounds})", DARK_BLUE)

        review = _review_report(state["task"], output, copywriter_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("COPYWRITER", "Reviewer approved the draft – no further changes", DARK_BLUE)
            break

        banner("COPYWRITER", f"Reviewer raised {len(comments)} comment(s) – revising", DARK_BLUE)
        output = _revise_report(state["task"], output, comments, copywriter_raw_results)
    else:
        banner("COPYWRITER", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", DARK_BLUE)

    title  = "Copywriter Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("COPYWRITER", f"Copywriter completed its task – invoking write_document tool", DARK_BLUE)
    copywriter_decision = "COPYWRITER_COMPLETED"

    banner("COPYWRITER", f"Decision → {copywriter_decision.upper()}", DARK_BLUE)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Copywriter Analysis:  → {summary}",f"[Copywriter] : route → {copywriter_decision}"]

    return {
        "copywriter_decision": copywriter_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }