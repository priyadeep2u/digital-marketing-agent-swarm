from shared import AgentState, banner, TEAL, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

DESIGN_AGENT_SYSTEM = """You are a Marketing Design Agent.

Create detailed design briefs for marketing assets.

Provide:
  • Layout recommendations
  • Visual hierarchy
  • Image concepts
  • Typography guidance
  • Branding considerations
  • Creative specifications

Focus on conversion-oriented design.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous Marketing Design Reviewer.
You will be given the original creative request or campaign source materials,
and the design brief that was written from them.
Review the design brief for:
  • Strategic alignment — flag any design recommendation in the brief that is not supported by, or
    contradicts, the supplied source materials or branding requirements
  • Completeness against the required sections (Layout recommendations, Visual hierarchy,
    Image concepts, Typography guidance, Branding considerations, Creative specifications)
  • Focus on conversion-oriented design principles (e.g., clear CTAs, frictionless flow)
  • Clarity, logical structure, and correct Markdown formatting
  • Vague, non-actionable, or off-brand creative directions
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the brief is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Marketing Design Agent revising your own report based on
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


@traceable(name="design_agent")
def DESIGN_AGENT_node(state: AgentState) -> dict:
    design_agent_search_query = state["task"]
    design_agent_search_query+= " for Designer perspective"
    design_agent_raw_results = ddg_search(query=design_agent_search_query, max_results=8)   
    banner("DESIGN AGENT", f"Creates design briefs for → {state['task'][:200]}.", TEAL)
    response = llm.invoke([
        SystemMessage(content=DESIGN_AGENT_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Designer search results:\n{design_agent_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("DESIGN AGENT", f"Sending draft to reviewer (round {review_rounds})", TEAL)

        review = _review_report(state["task"], output, design_agent_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("DESIGN AGENT", "Reviewer approved the draft – no further changes", TEAL)
            break

        banner("DESIGN AGENT", f"Reviewer raised {len(comments)} comment(s) – revising", TEAL)
        output = _revise_report(state["task"], output, comments, design_agent_raw_results)
    else:
        banner("DESIGN AGENT", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", TEAL)

    title  = "Design Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("DESIGN AGENT", f"Design Agent completed its task – invoking write_document tool", TEAL)
    design_agent_decision = "DESIGN_AGENT_COMPLETED"

    banner("DESIGN AGENT", f"Decision → {design_agent_decision.upper()}", TEAL)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Design Agent Analysis:  → {summary}",f"[Design Agent] : route → {design_agent_decision}"]

    return {
        "design_agent_decision": design_agent_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }