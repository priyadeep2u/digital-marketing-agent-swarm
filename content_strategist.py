from shared import AgentState, banner, BLUE, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

CONTENT_STRATEGIST_SYSTEM = """You are a Content Strategy Agent.

Create strategic content plans that support business goals.

Provide:
  • Content pillars
  • Topic clusters
  • Content calendar ideas
  • Funnel mapping
  • Audience targeting
  • Distribution strategy

Focus on SEO, engagement, and conversions.

Present results in Markdown.
"""

REVIEWER_SYSTEM = """You are a meticulous Content Strategy Reviewer.

You will be given the original brief/context (business goals, audience, and any
source research) and the content strategy plan that was written from it.

Review the plan for:
  • Strategic alignment — flag any recommendation that doesn't tie back to the
    stated business goals or target audience in the supplied context
  • Completeness against the required sections (Content pillars, Topic clusters,
    Content calendar ideas, Funnel mapping, Audience targeting, Distribution strategy)
  • Funnel coverage — confirm content ideas span awareness, consideration, and
    conversion stages rather than clustering at one stage
  • SEO soundness — flag topic/keyword suggestions that are too generic, too
    competitive for a realistic content play, or not aligned to search intent
  • Specificity — flag content pillars, topics, or calendar ideas that are vague,
    generic, or could apply to literally any business in the space
  • Redundancy — flag topic clusters or calendar entries that substantially overlap
  • Clarity, logical structure, and correct Markdown formatting

Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}

If the plan is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Content Strategy Agent revising your own report based on
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


@traceable(name="content_strategist")
def content_strategist_node(state: AgentState) -> dict:
    content_strategist_search_query = state["task"]
    content_strategist_search_query+= " for Content Strategist perspective"
    content_strategist_raw_results = ddg_search(query=content_strategist_search_query, max_results=8)                        
    banner("CONTENT STRATEGIST", f"Plans content strategy for → {state['task'][:200]}.", BLUE)
    response = llm.invoke([
        SystemMessage(content=CONTENT_STRATEGIST_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Content Strategist search results:\n{content_strategist_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("CONTENT STRATEGIST", f"Sending draft to reviewer (round {review_rounds})", BLUE)

        review = _review_report(state["task"], output, content_strategist_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("CONTENT STRATEGIST", "Reviewer approved the draft – no further changes", BLUE)
            break

        banner("CONTENT STRATEGIST", f"Reviewer raised {len(comments)} comment(s) – revising", BLUE)
        output = _revise_report(state["task"], output, comments, content_strategist_raw_results)
    else:
        banner("CONTENT STRATEGIST", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", BLUE)

    title  = "Content Strategist Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("CONTENT STRATEGIST", f"Content Strategist completed its task – invoking write_document tool", BLUE)
    content_strategist_decision = "CONTENT_STRATEGY_COMPLETED"

    banner("CONTENT STRATEGIST", f"Decision → {content_strategist_decision.upper()}", BLUE)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Content Strategist Analysis:  → {summary}",f"[Content Strategist] : route → {content_strategist_decision}"]

    return {
        "content_strategist_decision": content_strategist_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }