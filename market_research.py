from shared import AgentState, banner, CYAN, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

MARKET_RESEARCHER_SYSTEM = """You are an expert Market Research Agent.

Given a business, product, or industry, produce a structured research report:

  • Market overview
  • Industry trends
  • Competitor landscape
  • Target audience insights
  • Opportunities and threats
  • Customer pain points
  • Market gaps

Be data-driven and cite relevant findings when available.

Use Markdown bullet points and sections.
"""

REVIEWER_SYSTEM = """You are a meticulous Market Research Reviewer.
You will be given the original business, product, or industry source materials,
and the market research report that was generated from them.
Review the report for:
  • Strategic alignment and accuracy — flag any claims, data points, or findings that are not supported by, or
    contradict, the supplied source materials or established industry facts
  • Completeness against the required sections (Market overview, Industry trends,
    Competitor landscape, Target audience insights, Opportunities and threats, Customer pain points, Market gaps)
  • Analytical rigor — ensure the report is data-driven and cites relevant findings rather than
    relying on broad assumptions
  • Clarity, logical structure, and correct Markdown formatting (using bullet points and sections)
  • Vague trends, generic audience profiles, or unsupported market gaps
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the report is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Market Research Agent revising your own report based on
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


@traceable(name="market_research")
def market_researcher_node(state: AgentState) -> dict:
    Market_Researcher_search_query = state["task"]
    Market_Researcher_search_query+= " for Market Researcher perspective"
    Market_Researcher_raw_results = ddg_search(query=Market_Researcher_search_query, max_results=8) 
    banner("MARKET RESEARCHER", f"Plans marketing research for → {state['task'][:200]}.", CYAN)
    response = llm.invoke([
        SystemMessage(content=MARKET_RESEARCHER_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Market Researcher search results:\n{Market_Researcher_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("MARKET RESEARCHER", f"Sending draft to reviewer (round {review_rounds})", CYAN)

        review = _review_report(state["task"], output, Market_Researcher_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("MARKET RESEARCHER", "Reviewer approved the draft – no further changes", CYAN)
            break

        banner("MARKET RESEARCHER", f"Reviewer raised {len(comments)} comment(s) – revising", CYAN)
        output = _revise_report(state["task"], output, comments, Market_Researcher_raw_results)
    else:
        banner("MARKET RESEARCHER", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", CYAN)

    title  = "Marketing Research Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("MARKET RESEARCHER", f"Market Researcher completed its task – invoking write_document tool", CYAN)
    market_researcher_decision = "MARKET_RESEARCH_COMPLETED"

    banner("MARKET RESEARCHER", f"Decision → {market_researcher_decision.upper()}", CYAN)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Market Researcher Analysis:  → {summary}",f"[Market Researcher] : route → {market_researcher_decision}"]

    return {
        "market_researcher_decision": market_researcher_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }