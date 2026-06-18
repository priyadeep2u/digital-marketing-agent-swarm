from shared import AgentState, banner, YELLOW, write_tool, llm, json, ddg_search
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

MARKETING_DIRECTOR_SYSTEM = """You are the Marketing Director Agent. If the user asks about anything unrelated to a digital marketing task, respond: "I can only assist with digital marketing tasks."

Your role is to lead and coordinate a team of digital marketing specialists. If Reporting Agent reports are available, then mark the digital marketing task as finished.

Responsibilities:
  • Understand client goals and business objectives
  • Break projects into actionable tasks
  • Delegate work to specialized agents
  • Review outputs for quality and consistency
  • Build unified marketing strategies
  • Deliver final recommendations and reports

Always think strategically and focus on business outcomes.

Output:
  • Executive summaries
  • Marketing strategy documents
  • Campaign roadmaps
  • Action plans

Use clear Markdown formatting.
"""
MARKETING_KEYWORDS = {
    "marketing", "campaign", "brand", "seo", "ads", "advertising",
    "social media", "content", "email", "lead", "conversion", "funnel",
    "copywriting", "digital marketing", "google ads", "meta ads",
    "linkedin", "influencer", "market research", "competitor", "keyword",
    "ppc", "cro", "outreach", "product launch", "promotion", "audience"
}

def is_marketing_task(task: str) -> bool:
    task_lower = task.lower()
    return any(kw in task_lower for kw in MARKETING_KEYWORDS)

@traceable(name="marketing_director")
def marketing_director_node(state: AgentState) -> dict:
    marketing_director_search_query = state["task"]
    marketing_director_search_query+= " for Marketing Director perspective"
    marketing_director_raw_results = ddg_search(query=marketing_director_search_query, max_results=8)        
    task = state["task"]

    banner("MARKETING DIRECTOR", f"Plans marketing research for → {task[:200]}.", YELLOW)

    msgs        = list(state.get("messages", []))
    output_file = []
    reporting_done = state.get("reporting_decision") == "REPORTING_COMPLETED"

    relevant = is_marketing_task(task)

    if not relevant:
        relevance_response = llm.invoke([
            SystemMessage(content=(
                "You are a strict gatekeeper. Answer ONLY 'YES' or 'NO'.\n"
                "A task is a digital marketing task ONLY if it explicitly involves: "
                "SEO, ads, campaigns, branding, content marketing, social media marketing, "
                "email marketing, lead generation, market research for a product/brand, "
                "copywriting for marketing, or similar. "
                "General tech, AI, science, or news questions are NOT marketing tasks."
            )),
            HumanMessage(content=f"Is this a digital marketing task? Task: {task}"),
        ])
        relevant = "yes" in relevance_response.content.strip().lower()

    if not relevant or reporting_done:
        banner("MARKETING DIRECTOR", "Task is not a digital marketing request → FINISH", YELLOW)
        marketing_director_decision = "FINISH"
    else:
        response = llm.invoke([
            SystemMessage(content=MARKETING_DIRECTOR_SYSTEM),
            HumanMessage(content=(
                f"Task: {state['task']}\n\n"
                f"Marketing Director search results:\n{marketing_director_raw_results}\n\n"
            )),
        ])
        output = response.content.strip()
        title  = "Marketing Director Agent"

        tool_result_raw = write_tool._run(title=title, content=output)
        tool_result     = json.loads(tool_result_raw)
        output_file     = [tool_result["file"]]
        summary = output[:300].rsplit('\n', 1)[0] + "…" if len(output) > 300 else output
        msgs.append(f" Marketing Director Analysis:  → {summary}")
        banner("MARKETING DIRECTOR", "Marketing Director completed – invoking write_document tool", YELLOW)
        marketing_director_decision = "RESEARCH_ROUTER"

    banner("MARKETING DIRECTOR", f"Decision → {marketing_director_decision}", YELLOW)
    msgs.append(f"[Marketing Director] : route → {marketing_director_decision}")

    return {
        "marketing_director_decision": marketing_director_decision,
        "output_file":                 output_file,
        "messages":                    msgs,
    }