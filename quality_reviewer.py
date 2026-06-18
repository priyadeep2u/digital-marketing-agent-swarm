from shared import AgentState, banner, GOLD, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

QUALITY_REVIEWER_SYSTEM = """You are a Quality Assurance Agent.

Review outputs produced by other agents.

Evaluate:
  • Accuracy
  • Completeness
  • Consistency
  • Clarity
  • Brand alignment
  • Strategic value

Identify issues and provide improvement recommendations.

Use concise Markdown feedback.
"""

REVIEWER_SYSTEM = """You are a meticulous Quality Assurance Meta-Reviewer.
You will be given the original source materials, the initial agent output that was evaluated,
and the QA review report that was generated from them.
Review the QA report for:
  • Accuracy and alignment — flag any feedback or evaluation that is not supported by, or
    contradicts, the original source materials or misinterprets the initial agent's output
  • Completeness against the required sections (Evaluation of Accuracy, Completeness, Consistency,
    Clarity, Brand alignment, and Strategic value; Identified issues; Improvement recommendations)
  • Constructive value — ensure the feedback provides clear, actionable improvement recommendations
    rather than simply listing flaws without solutions
  • Clarity, logical structure, and correct Markdown formatting
  • Unjustified criticisms, overly vague feedback, or subjective nitpicking
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the QA report is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Quality Assurance Agent revising your own report based on
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


@traceable(name="quality_reviewer")
def QUALITY_REVIEWER_node(state: AgentState) -> dict:
    Quality_Reviewer_search_query = state["task"]
    Quality_Reviewer_search_query+= " for Quality Reviewer perspective"
    Quality_Reviewer_raw_results = ddg_search(query=Quality_Reviewer_search_query, max_results=8) 
    banner("QUALITY REVIEWER", f"Reviews outputs for → {state['task'][:200]}.", GOLD)
    response = llm.invoke([
        SystemMessage(content=QUALITY_REVIEWER_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Quality Reviewer search results:\n{Quality_Reviewer_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("QUALITY REVIEWER", f"Sending draft to reviewer (round {review_rounds})", GOLD)

        review = _review_report(state["task"], output, Quality_Reviewer_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("QUALITY REVIEWER", "Reviewer approved the draft – no further changes", GOLD)
            break

        banner("QUALITY REVIEWER", f"Reviewer raised {len(comments)} comment(s) – revising", GOLD)
        output = _revise_report(state["task"], output, comments, Quality_Reviewer_raw_results)
    else:
        banner("QUALITY REVIEWER", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", GOLD)

    title  = "Quality Reviewer Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("QUALITY REVIEWER", f"Quality Reviewer completed its task – invoking write_document tool", GOLD)
    quality_reviewer_decision = "QUALITY_REVIEWER_COMPLETED"

    banner("QUALITY REVIEWER", f"Decision → {quality_reviewer_decision.upper()}", GOLD)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Quality Reviewer Analysis:  → {summary}",f"[Quality Reviewer] : route → {quality_reviewer_decision}"]

    return {
        "quality_reviewer_decision": quality_reviewer_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }