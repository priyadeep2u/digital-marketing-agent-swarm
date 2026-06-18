from shared import AgentState, banner, DARK_CYAN, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

EMAIL_MARKETER_SYSTEM = """You are an Email Marketing Agent.

Create email campaigns that drive engagement and conversions.

Provide:
  • Subject lines
  • Preview text
  • Email sequences
  • CTA recommendations
  • Segmentation suggestions

Optimize for:
  • Open rates
  • Click-through rates
  • Conversions

Output in Markdown.
"""

REVIEWER_SYSTEM = """You are a meticulous Email Marketing Reviewer.
You will be given the original campaign brief, target audience details, or promotional source materials,
and the email campaign that was written from them.
Review the email campaign for:
  • Strategic alignment — flag any messaging in the campaign that is not supported by, or
    contradicts, the supplied source materials or target audience needs
  • Completeness against the required sections (Subject lines, Preview text,
    Email sequences, CTA recommendations, Segmentation suggestions)
  • Optimization for engagement — ensure best practices are used to drive open rates (e.g., avoiding spam triggers),
    click-through rates (e.g., clear, prominent CTAs), and conversions
  • Clarity, logical structure, and correct Markdown formatting
  • Spammy language, generic subject lines, or weak calls-to-action
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the campaign is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Email Marketing Agent revising your own report based on
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


@traceable(name="email_marketer")
def EMAIL_MARKETER_node(state: AgentState) -> dict:
    email_marketer_search_query = state["task"]
    email_marketer_search_query+= " for Email Marketer perspective"
    email_marketer_raw_results = ddg_search(query=email_marketer_search_query, max_results=8)    
    banner("EMAIL MARKETER", f"Creates email campaigns for → {state['task'][:200]}.", DARK_CYAN)
    response = llm.invoke([
        SystemMessage(content=EMAIL_MARKETER_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Email Marketer search results:\n{email_marketer_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("EMAIL MARKETER", f"Sending draft to reviewer (round {review_rounds})", DARK_CYAN)

        review = _review_report(state["task"], output, email_marketer_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("EMAIL MARKETER", "Reviewer approved the draft – no further changes", DARK_CYAN)
            break

        banner("EMAIL MARKETER", f"Reviewer raised {len(comments)} comment(s) – revising", DARK_CYAN)
        output = _revise_report(state["task"], output, comments, email_marketer_raw_results)
    else:
        banner("EMAIL MARKETER", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", DARK_CYAN)

    title  = "Email Marketer Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("EMAIL MARKETER", f"Email Marketer completed its task – invoking write_document tool", DARK_CYAN)
    email_marketer_decision = "EMAIL_MARKETER_COMPLETED"

    banner("EMAIL MARKETER", f"Decision → {email_marketer_decision.upper()}", DARK_CYAN)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Email Marketer Analysis:  → {summary}",f"[Email Marketer] : route → {email_marketer_decision}"]

    return {
        "email_marketer_decision": email_marketer_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }