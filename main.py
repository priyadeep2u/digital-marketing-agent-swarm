import os
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from shared import AgentState, OLLAMA_MODEL, GREEN, OLLAMA_REVIEWER_MODEL, BOLD, RESET
from langsmith import traceable

# Enable tracing
os.environ["LANGSMITH_TRACING"] = "true"

load_dotenv()

from competitor_analyst import competitor_analyst_node
from analytics import ANALYTICS_node
from content_strategist import content_strategist_node
from copywriter import COPYWRITER_node
from content_writer import CONTENT_WRITER_node
from creative_director import CREATIVE_DIRECTOR_node
from cro_specialist import CRO_SPECIALIST_node
from design_agent import DESIGN_AGENT_node
from email_marketer import EMAIL_MARKETER_node
from google_ads import GOOGLE_ADS_node
from keyword_research import KEYWORD_RESEARCH_node
from lead_generation import LEAD_GENERATION_node
from linkedin_specialist import LINKEDIN_SPECIALIST_node
from market_research import market_researcher_node
from marketing_director import marketing_director_node
from meta_ads import META_ADS_node
from outreach_agent import OUTREACH_AGENT_node
from paid_ads import PAID_ADS_node
from quality_reviewer import QUALITY_REVIEWER_node
from reporting import REPORTING_node
from seo_specialist import SEO_Specialist_node
from social_media import SOCIAL_MEDIA_MANAGER_node
from trend_analyst import trend_analyst_node


# ── Gate nodes ────────────────────────────────────────────────────────────────

def research_gate_node(state: AgentState) -> dict:
    """Fan-in: waits for market_researcher + trend_analyst."""
    return {}

def pre_meta_gate_node(state: AgentState) -> dict:
    """Fan-in: waits for keyword_research + lead_generation + google_ads
       before meta_ads runs — prevents meta_ads firing 3 times."""
    return {}

def final_gate_node(state: AgentState) -> dict:
    """Fan-in: waits for analytics + linkedin_specialist + email_marketer
       before reporting."""
    return {}

def route_marketing_director(state: AgentState) -> list | str:
    if state.get("marketing_director_decision") == "FINISH":
        return END
    return ["market_researcher", "trend_analyst"]   # parallel fan-out

def build_graph() -> StateGraph:
    wf = StateGraph(AgentState)

    # Agent nodes
    wf.add_node("competitor_analyst",   competitor_analyst_node)
    wf.add_node("analytics",            ANALYTICS_node)
    wf.add_node("content_strategist",   content_strategist_node)
    wf.add_node("copywriter",           COPYWRITER_node)
    wf.add_node("content_writer",       CONTENT_WRITER_node)
    wf.add_node("creative_director",    CREATIVE_DIRECTOR_node)
    wf.add_node("cro",                  CRO_SPECIALIST_node)
    wf.add_node("design_agent",         DESIGN_AGENT_node)
    wf.add_node("email_marketer",       EMAIL_MARKETER_node)
    wf.add_node("google_ads",           GOOGLE_ADS_node)
    wf.add_node("keyword_research",     KEYWORD_RESEARCH_node)
    wf.add_node("lead_generation",      LEAD_GENERATION_node)
    wf.add_node("linkedin_specialist",  LINKEDIN_SPECIALIST_node)
    wf.add_node("market_researcher",    market_researcher_node)
    wf.add_node("marketing_director",   marketing_director_node)
    wf.add_node("meta_ads",             META_ADS_node)
    wf.add_node("outreach",             OUTREACH_AGENT_node)
    wf.add_node("paid_ads",             PAID_ADS_node)
    wf.add_node("quality_reviewer",     QUALITY_REVIEWER_node)
    wf.add_node("reporting",            REPORTING_node)
    wf.add_node("seo_specialist",       SEO_Specialist_node)
    wf.add_node("social_media_manager", SOCIAL_MEDIA_MANAGER_node)
    wf.add_node("trend_analyst",        trend_analyst_node)

    wf.add_edge(START, "marketing_director")
    wf.add_conditional_edges("marketing_director", route_marketing_director,
        {END: END, "market_researcher": "market_researcher",
                   "trend_analyst":     "trend_analyst"})

    wf.add_edge(["market_researcher","trend_analyst"],  "competitor_analyst")
    wf.add_edge("competitor_analyst", "content_strategist")
    wf.add_edge("content_strategist", "seo_specialist")
    wf.add_edge("content_strategist", "paid_ads")
    wf.add_edge("content_strategist", "lead_generation")
    wf.add_edge("paid_ads",         "google_ads")
    wf.add_edge("google_ads",       "meta_ads")
    wf.add_edge(["seo_specialist","lead_generation","meta_ads"],    "keyword_research")
    wf.add_edge("keyword_research",         "copywriter")
    wf.add_edge("keyword_research",         "outreach")
    wf.add_edge("copywriter",       "content_writer")
    wf.add_edge("content_writer",   "cro")
    wf.add_edge("cro",              "social_media_manager")
    wf.add_edge("social_media_manager", "linkedin_specialist")
    wf.add_edge("social_media_manager", "creative_director")
    wf.add_edge("creative_director", "design_agent")
    wf.add_edge("design_agent",      "quality_reviewer")
    wf.add_edge("quality_reviewer",  "analytics")
    wf.add_edge("outreach",         "email_marketer")
    wf.add_edge(["analytics","email_marketer","linkedin_specialist"],     "reporting")
    wf.add_edge("reporting", "marketing_director")

    return wf.compile()

@traceable(name="Digital Marketing Agent Swarm")
def run(msgs: list, query: str) -> dict:
    graph = build_graph()

    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("output/graph.png", "wb") as f:
        f.write(png_bytes)
    print("\nGraph saved to graph.png")

    print(f"\n{'═'*65}")
    print(f"{BOLD}  MULTI-AGENT PIPELINE  –  Ollama · LangGraph · LangSmith{RESET}")
    print(f"{'═'*65}")
    print(f"  Model         : {OLLAMA_MODEL}")
    print(f"  Reviewer Model: {OLLAMA_REVIEWER_MODEL}")
    print(f"  Task          : {msgs}")
    print(f"{'═'*65}\n")

    initial_state: AgentState = {
        "competitor_analyst_decision":  "",
        "content_strategist_decision":  "",
        "content_writer_decision":      "",
        "copywriter_decision":          "",
        "creative_director_decision":   "",
        "cro_decision":                 "",
        "design_agent_decision":        "",
        "email_marketer_decision":      "",
        "google_ads_decision":          "",
        "keyword_research_decision":    "",
        "lead_generation_decision":     "",
        "linkedin_specialist_decision": "",
        "market_researcher_decision":   "",
        "marketing_director_decision":  "",
        "meta_ads_decision":            "",
        "outreach_decision":            "",
        "paid_ads_decision":            "",
        "quality_reviewer_decision":    "",
        "reporting_decision":           "",
        "seo_specialist_decision":      "",
        "social_media_decision":        "",
        "trend_analyst_decision":       "",
        "output_file":                  [],
        "messages":                     msgs,
        "task":                         query,
    }

    final_state = graph.invoke(initial_state)

    print(f"\n{'═'*65}")
    print(f"{BOLD}{GREEN}  PIPELINE COMPLETE{RESET}")
    print(f"{'═'*65}")
    print(f"  Output files:")
    for f in final_state.get("output_file", []):
        print(f"    • {f}")
    print(f"\n  Audit trail:")
    for msg in final_state.get("messages", []):
        print(f"    • {msg}")
    print(f"{'═'*65}\n")
    return final_state


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Please create a digital marketing idea for a small online shopping company "
        "selling exclusive perfume in India"
    )
    msgs = [f" Client Requirement:  → {query}"]
    run(msgs, query)