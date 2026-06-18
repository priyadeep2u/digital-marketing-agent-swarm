from shared import AgentState, banner, MAGENTA, write_tool, llm, json, ddg_search, llm_reviewer 
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

SEO_SPECIALIST_SYSTEM = """You are an expert SEO Specialist Agent.

Analyze search opportunities and produce:

  • Primary keywords
  • Secondary keywords
  • Long-tail opportunities
  • Search intent classification
  • Topic clusters
  • Content gaps
  • Internal linking suggestions
  • SEO recommendations

Prioritize keywords based on relevance, competition, and business value.

Output in structured Markdown.
"""

REVIEWER_SYSTEM = """You are a meticulous SEO Specialist Reviewer.
You will be given the original seed topics, website context, or target market materials,
and the SEO strategy report that was generated from them.
Review the report for:
  • Strategic alignment — flag any keywords, clusters, or recommendations that are not supported by, or
    contradict, the supplied source materials, target audience, or search intent
  • Completeness against the required sections (Primary keywords, Secondary keywords,
    Long-tail opportunities, Search intent classification, Topic clusters, Content gaps,
    Internal linking suggestions, SEO recommendations)
  • Prioritization — ensure recommendations and keywords are logically prioritized based on
    relevance, competition, and business value rather than just raw search volume
  • Clarity, logical structure, and correct Markdown formatting
  • Inaccurate intent classification, generic content gaps, or unrealistic/spammy SEO recommendations
Respond ONLY with a JSON object, no preamble, no code fences:
{"approved": <true|false>, "comments": ["comment 1", "comment 2", ...]}
If the report is strong and needs no changes, return "approved": true and "comments": [].
Only raise comments that are specific, actionable, and clearly valid — do not nitpick
phrasing or personal style preferences.
"""

REVISION_SYSTEM = """You are the SEO Specialist Agent revising your own report based on
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


@traceable(name="seo_specialist")
def SEO_Specialist_node(state: AgentState) -> dict:
    SEO_Specialist_search_query = state["task"]
    SEO_Specialist_search_query+= " for SEO Specialist perspective"
    SEO_Specialist_raw_results = ddg_search(query=SEO_Specialist_search_query, max_results=8)  
    banner("SEO SPECIALIST", f"Analyzes SEO opportunities for → {state['task'][:200]}.", MAGENTA)
    response = llm.invoke([
        SystemMessage(content=SEO_SPECIALIST_SYSTEM),
        HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"SEO Specialist search results:\n{SEO_Specialist_raw_results}\n\n"
            )
        ),
      ])

    output = response.content.strip()

    review_rounds = 0
    while review_rounds < MAX_REVIEW_ROUNDS:
        review_rounds += 1
        banner("SEO SPECIALIST", f"Sending draft to reviewer (round {review_rounds})", MAGENTA)

        review = _review_report(state["task"], output, SEO_Specialist_raw_results)
        comments = review.get("comments", [])
        approved = review.get("approved", False)

        if approved or not comments:
            banner("SEO SPECIALIST", "Reviewer approved the draft – no further changes", MAGENTA)
            break

        banner("SEO SPECIALIST", f"Reviewer raised {len(comments)} comment(s) – revising", MAGENTA)
        output = _revise_report(state["task"], output, comments, SEO_Specialist_raw_results)
    else:
        banner("SEO SPECIALIST", f"Reached max review rounds ({MAX_REVIEW_ROUNDS}) – finalizing", MAGENTA)

    title  = "SEO specialist Agent"

    tool_result_raw = write_tool._run(title=title, content=output)
    tool_result = json.loads(tool_result_raw)
    banner("SEO SPECIALIST", f"SEO Specialist completed its task – invoking write_document tool", MAGENTA)
    seo_specialist_decision = "SEO_ANALYSIS_COMPLETED"

    banner("SEO SPECIALIST", f"Decision → {seo_specialist_decision.upper()}", MAGENTA)

    summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output

    msgs = [f" SEO Specialist Analysis:  → {summary}",f"[SEO Specialist] : route → {seo_specialist_decision}"]

    return {
        "seo_specialist_decision": seo_specialist_decision,
        "output_file": [tool_result["file"]],
        "messages":                    msgs,
    }