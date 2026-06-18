from shared import AgentState, banner, MAGENTA, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

LEAD_GENERATION_SYSTEM = """You are a Lead Generation Agent.

Identify opportunities to generate qualified leads.

Provide:
  • Ideal customer profiles
  • Prospecting strategies
  • Outreach recommendations
  • Funnel improvements
  • Lead magnets
  • Qualification criteria

Focus on scalable acquisition strategies.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous Lead Generation Reviewer.
You will be given the original business goals, product details, or market research materials,
and the lead generation strategy that was generated from them.
Review the strategy for:
  • Strategic alignment — flag any profiles, strategies, or criteria that are not supported by, or
    contradict, the supplied source materials or target market realities
  • Completeness against the required sections (Ideal customer profiles, Prospecting strategies,
    Outreach recommendations, Funnel improvements, Lead magnets, Qualification criteria)
  • Focus on scalability — ensure the acquisition strategies are practical, scalable, and
    optimized for capturing highly qualified leads rather than just volume
  • Clarity, logical structure, and correct Markdown formatting
  • Unrealistic acquisition tactics, poorly defined qualification criteria, or generic lead magnets
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the strategy is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Lead Generation Agent revising your own report based on
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


@traceable(name="lead_generation")
def LEAD_GENERATION_node(state: AgentState) -> dict:
    lead_generation_search_query = state["task"]
    lead_generation_search_query+= " for Lead Generation perspective"
    lead_generation_raw_results = ddg_search(query=lead_generation_search_query, max_results=8)   
    banner("LEAD GENERATION", f"Generates leads for → {state['task'][:200]}.", MAGENTA)
    response = llm.invoke([
        SystemMessage(content=LEAD_GENERATION_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Lead Generation search results:\n{lead_generation_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("LEAD GENERATION", f"Sending draft to reviewer (round {review_rounds})", MAGENTA)

        review = _review_report(state["task"], output, lead_generation_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("LEAD GENERATION", "Reviewer approved the draft – no further changes", MAGENTA)
            break

        banner("LEAD GENERATION", f"Reviewer raised {len(comments)} comment(s) – revising", MAGENTA)
        output = _revise_report(state["task"], output, comments, lead_generation_raw_results)
    else:
        banner("LEAD GENERATION", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", MAGENTA)

    title  = "Lead Generation Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("LEAD GENERATION", f"Lead Generation completed its task – invoking write_document tool", MAGENTA)
    lead_generation_decision = "LEAD_GENERATION_COMPLETED"

    banner("LEAD GENERATION", f"Decision → {lead_generation_decision.upper()}", MAGENTA)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Lead Generation Analysis:  → {summary}",f"[Lead Generation] : route → {lead_generation_decision}"]

    return {
        "lead_generation_decision": lead_generation_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }