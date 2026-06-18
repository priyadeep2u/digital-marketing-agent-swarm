from shared import AgentState, banner, DARK_RED, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

KEYWORD_RESEARCH_SYSTEM = """You are a Keyword Research Agent.

Generate comprehensive keyword research including:

  • Primary keywords
  • Secondary keywords
  • Long-tail keywords
  • Question-based searches
  • Commercial intent keywords
  • Informational intent keywords
  • Cluster recommendations

Organize keywords by topic and intent.

Use Markdown tables when appropriate.
"""

REVIEWER_SYSTEM = """You are a meticulous Keyword Research Reviewer.
You will be given the original seed topics, business brief, or target website materials,
and the keyword research report that was generated from them.
Review the report for:
  • Strategic alignment — flag any keywords or clusters that are irrelevant, off-topic, or
    do not align with the supplied seed materials and target audience
  • Completeness against the required sections (Primary keywords, Secondary keywords,
    Long-tail keywords, Question-based searches, Commercial intent keywords,
    Informational intent keywords, Cluster recommendations)
  • Organization and formatting — ensure keywords are logically organized by topic and intent,
    and that Markdown tables are used appropriately where applicable
  • Redundant keywords, inaccurate intent classification, or weak cluster groupings
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the report is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Keyword Research Agent revising your own report based on
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


@traceable(name="keyword_research")
def KEYWORD_RESEARCH_node(state: AgentState) -> dict:
    keyword_research_search_query = state["task"]
    keyword_research_search_query+= " for Keyword Research perspective"
    keyword_research_raw_results = ddg_search(query=keyword_research_search_query, max_results=8)   
    banner("KEYWORD RESEARCH", f"Generates keyword research for → {state['task'][:200]}.", DARK_RED)
    response = llm.invoke([
        SystemMessage(content=KEYWORD_RESEARCH_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Keyword Research search results:\n{keyword_research_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("KEYWORD RESEARCH", f"Sending draft to reviewer (round {review_rounds})", DARK_RED)

        review = _review_report(state["task"], output, keyword_research_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("KEYWORD RESEARCH", "Reviewer approved the draft – no further changes", DARK_RED)
            break

        banner("KEYWORD RESEARCH", f"Reviewer raised {len(comments)} comment(s) – revising", DARK_RED)
        output = _revise_report(state["task"], output, comments, keyword_research_raw_results)
    else:
        banner("KEYWORD RESEARCH", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", DARK_RED)

    title  = "Keyword Research Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("KEYWORD RESEARCH", f"Keyword Research completed its task – invoking write_document tool", DARK_RED)
    keyword_research_decision = "KEYWORD_RESEARCH_COMPLETED"

    banner("KEYWORD RESEARCH", f"Decision → {keyword_research_decision.upper()}", DARK_RED)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Keyword Research Analysis:  → {summary}",f"[Keyword Research] : route → {keyword_research_decision}"]

    return {
        "keyword_research_decision": keyword_research_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }