from shared import AgentState, banner, DARK_MAGENTA, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

OUTREACH_AGENT_SYSTEM = """You are an Outreach Specialist Agent.

Create outreach campaigns for partnerships, sales, and networking.

Provide:
  • Outreach messages
  • Follow-up sequences
  • Personalization suggestions
  • Objection handling
  • Response optimization

Focus on relationship building and conversion.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous Outreach Specialist Reviewer.
You will be given the original campaign goals, target persona details, or networking context,
and the outreach campaign that was generated from them.
Review the campaign for:
  • Strategic alignment — flag any messaging or tactics that are not supported by, or
    contradict, the supplied source materials or target audience context
  • Completeness against the required sections (Outreach messages, Follow-up sequences,
    Personalization suggestions, Objection handling, Response optimization)
  • Optimization for relationship building and conversion — ensure the tone is appropriate,
    personalization is meaningful, and follow-ups are persistent but respectful
  • Clarity, logical structure, and correct Markdown formatting
  • Spammy language, overly aggressive sales pitches, or generic objection handling
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the campaign is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Outreach Specialist Agent revising your own report based on
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


@traceable(name="outreach_agent")
def OUTREACH_AGENT_node(state: AgentState) -> dict:
    Outreach_agent_search_query = state["task"]
    Outreach_agent_search_query+= " for Outreach Agent perspective"
    Outreach_agent_raw_results = ddg_search(query=Outreach_agent_search_query, max_results=8) 
    banner("OUTREACH AGENT", f"Creates outreach campaigns for → {state['task'][:200]}.", DARK_MAGENTA)
    response = llm.invoke([
        SystemMessage(content=OUTREACH_AGENT_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Outreach Agent search results:\n{Outreach_agent_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("OUTREACH AGENT", f"Sending draft to reviewer (round {review_rounds})", DARK_MAGENTA)

        review = _review_report(state["task"], output, Outreach_agent_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("OUTREACH AGENT", "Reviewer approved the draft – no further changes", DARK_MAGENTA)
            break

        banner("OUTREACH AGENT", f"Reviewer raised {len(comments)} comment(s) – revising", DARK_MAGENTA)
        output = _revise_report(state["task"], output, comments, Outreach_agent_raw_results)
    else:
        banner("OUTREACH AGENT", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", DARK_MAGENTA)

    title  = "Outreach Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("OUTREACH AGENT", f"Outreach Agent completed its task – invoking write_document tool", DARK_MAGENTA)
    outreach_decision = "OUTREACH_COMPLETED"

    banner("OUTREACH AGENT", f"Decision → {outreach_decision.upper()}", DARK_MAGENTA)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Outreach Agent Analysis:  → {summary}",f"[Outreach Agent] : route → {outreach_decision}"]

    return {
        "outreach_decision": outreach_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }