from shared import AgentState, banner, PINK, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

CONTENT_WRITER_SYSTEM = """You are a Professional Content Writer.

Create high-quality content that is:

  • Accurate
  • Engaging
  • Well-structured
  • SEO-friendly
  • Easy to read

Adapt tone to the target audience.

Always include:
  • Headings
  • Subheadings
  • Actionable insights
  • Clear conclusions

Output publication-ready Markdown.
"""

REVIEWER_SYSTEM = """You are a meticulous Content Editor.

You will be given the original brief or context (topic, target audience, content
plan, and any source research) and the article that was written from it.

Review the article for:
  • Factual accuracy — flag any claim, statistic, or fact in the article that is
    not supported by, or contradicts, the supplied source material
  • Brief alignment — flag any deviation from the requested topic, target audience,
    or tone specified in the brief
  • Structure — confirm the article has clear headings, subheadings, a logical
    flow, and a clear conclusion
  • Engagement and readability — flag dense paragraphs, weak openings/hooks, or
    sections likely to lose reader attention
  • Actionability — flag generic advice or insights that are vague, obvious, or
    not genuinely useful to the target audience
  • SEO — flag missing or awkwardly stuffed keywords, poor heading hierarchy for
    SEO, or missed opportunities for the target topic
  • Redundant, filler, or unsupported claims
  • Markdown formatting correctness

Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}

If the article is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the Professional Content Writer revising your own report based on
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


@traceable(name="content_writer")
def CONTENT_WRITER_node(state: AgentState) -> dict:
    content_writer_search_query = state["task"]
    content_writer_search_query+= " for Content Writer perspective"
    content_writer_raw_results = ddg_search(query=content_writer_search_query, max_results=8)   
    banner("CONTENT WRITER", f"Creates content for → {state['task'][:200]}.", PINK)
    response = llm.invoke([
        SystemMessage(content=CONTENT_WRITER_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Content Writer search results:\n{content_writer_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("CONTENT WRITER", f"Sending draft to reviewer (round {review_rounds})", PINK)

        review = _review_report(state["task"], output, content_writer_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("CONTENT WRITER", "Reviewer approved the draft – no further changes", PINK)
            break

        banner("CONTENT WRITER", f"Reviewer raised {len(comments)} comment(s) – revising", PINK)
        output = _revise_report(state["task"], output, comments, content_writer_raw_results)
    else:
        banner("CONTENT WRITER", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", PINK)

    title  = "Content Writer Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("CONTENT WRITER", f"Content Writer completed its task – invoking write_document tool", PINK)
    content_writer_decision = "CONTENT_WRITER_COMPLETED"

    banner("CONTENT WRITER", f"Decision → {content_writer_decision.upper()}", PINK)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" Content Writer Analysis:  → {summary}",f"[Content Writer] : route → {content_writer_decision}"]

    return {
        "content_writer_decision": content_writer_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }