from shared import AgentState, banner, LIME, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

CREATIVE_DIRECTOR_SYSTEM = """You are a Creative Director Agent.

Develop creative concepts for marketing campaigns.

Provide:
  • Campaign themes
  • Visual concepts
  • Messaging frameworks
  • Brand alignment recommendations
  • Creative direction

Focus on consistency, memorability, and audience impact.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous Creative Director Reviewer.

You will be given the original brief or context (business goals, target audience,
brand guidelines, and any prior strategy/research) and the creative concept that
was developed from it.

Review the concept for:
  • Completeness against the required sections (Campaign themes, Visual concepts,
    Messaging frameworks, Brand alignment recommendations, Creative direction)
  • Brand alignment — flag any visual concept, theme, or messaging direction that
    contradicts or strays from the supplied brand guidelines
  • Audience fit — flag concepts unlikely to resonate with, or appropriate for,
    the stated target audience
  • Strategic alignment — flag creative direction that doesn't tie back to the
    business goals or upstream strategy provided in the brief
  • Distinctiveness — flag themes or concepts that feel generic, derivative, or
    indistinguishable from category conventions rather than memorable
  • Internal consistency — flag mismatches between the stated theme, the visual
    concepts, and the messaging framework (i.e. they should feel like one campaign,
    not disconnected pieces)
  • Feasibility — flag visual concepts or directions that are vague enough to be
    unactionable for a design/production team
  • Markdown formatting correctness

Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}

If the concept is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Creative Director Agent revising your own report based on
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


@traceable(name="creative_director")
def CREATIVE_DIRECTOR_node(state: AgentState) -> dict:
    creative_director_search_query = state["task"]
    creative_director_search_query+= " for Creative Director perspective"
    creative_director_raw_results = ddg_search(query=creative_director_search_query, max_results=8)
    banner("CREATIVE DIRECTOR", f"Develops creative concepts for → {state['task'][:200]}.", LIME)
    response = llm.invoke([
        SystemMessage(content=CREATIVE_DIRECTOR_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Creative Director search results:\n{creative_director_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("CREATIVE DIRECTOR", f"Sending draft to reviewer (round {review_rounds})", LIME)

        review = _review_report(state["task"], output, creative_director_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("CREATIVE DIRECTOR", "Reviewer approved the draft – no further changes", LIME)
            break

        banner("CREATIVE DIRECTOR", f"Reviewer raised {len(comments)} comment(s) – revising", LIME)
        output = _revise_report(state["task"], output, comments, creative_director_raw_results)
    else:
        banner("CREATIVE DIRECTOR", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", LIME)

    title  = "Creative Director Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("CREATIVE DIRECTOR", f"Creative Director completed its task – invoking write_document tool", LIME)
    creative_director_decision = "CREATIVE_DIRECTOR_COMPLETED"

    banner("CREATIVE DIRECTOR", f"Decision → {creative_director_decision.upper()}", LIME)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Creative Director Analysis:  → {summary}",f"[Creative Director] : route → {creative_director_decision}"]

    return {
        "creative_director_decision": creative_director_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }