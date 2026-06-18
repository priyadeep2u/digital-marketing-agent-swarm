from shared import AgentState, banner, PURPLE, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

LINKEDIN_SPECIALIST_SYSTEM = """You are a LinkedIn Growth Agent.

Create content and strategies optimized for LinkedIn.

Provide:
  • Thought leadership posts
  • Industry insights
  • Engagement hooks
  • Comment strategies
  • Networking opportunities
  • Lead generation tactics

Focus on professional audiences and B2B growth.

Use Markdown formatting.
"""

REVIEWER_SYSTEM = """You are a meticulous LinkedIn Growth Reviewer.
You will be given the original brand guidelines, target audience details, or executive profiles,
and the LinkedIn strategy that was generated from them.
Review the strategy for:
  • Strategic alignment — flag any content or tactics that are not supported by, or
    contradict, the supplied source materials or B2B focus
  • Completeness against the required sections (Thought leadership posts, Industry insights,
    Engagement hooks, Comment strategies, Networking opportunities, Lead generation tactics)
  • Optimization for B2B growth — ensure the content appeals to professional audiences and
    drives meaningful engagement rather than superficial vanity metrics
  • Clarity, logical structure, and correct Markdown formatting
  • Unprofessional language, generic engagement bait, or ineffective networking tactics
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the strategy is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the LinkedIn Growth Agent revising your own report based on
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


@traceable(name="linkedin_specialist")
def LINKEDIN_SPECIALIST_node(state: AgentState) -> dict:
    Linkedin_Specialist_search_query = state["task"]
    Linkedin_Specialist_search_query+= " for Linkedin Specialist perspective"
    Linkedin_Specialist_raw_results = ddg_search(query=Linkedin_Specialist_search_query, max_results=8)
    banner("LINKEDIN SPECIALIST", f"Creates LinkedIn content for → {state['task'][:200]}.", PURPLE)
    response = llm.invoke([
        SystemMessage(content=LINKEDIN_SPECIALIST_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Linkedin Specialist search results:\n{Linkedin_Specialist_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("LINKEDIN SPECIALIST", f"Sending draft to reviewer (round {review_rounds})", PURPLE)

        review = _review_report(state["task"], output, Linkedin_Specialist_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("LINKEDIN SPECIALIST", "Reviewer approved the draft – no further changes", PURPLE)
            break

        banner("LINKEDIN SPECIALIST", f"Reviewer raised {len(comments)} comment(s) – revising", PURPLE)
        output = _revise_report(state["task"], output, comments, Linkedin_Specialist_raw_results)
    else:
        banner("LINKEDIN SPECIALIST", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", PURPLE)

    title  = "Linkedin Specialist Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("LINKEDIN SPECIALIST", f"LinkedIn Specialist completed its task – invoking write_document tool", PURPLE)
    linkedin_specialist_decision = "LINKEDIN_SPECIALIST_COMPLETED"

    banner("LINKEDIN SPECIALIST", f"Decision → {linkedin_specialist_decision.upper()}", PURPLE)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" LinkedIn Specialist Analysis:  → {summary}",f"[LinkedIn Specialist] : route → {linkedin_specialist_decision}"]

    return {
        "linkedin_specialist_decision": linkedin_specialist_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }