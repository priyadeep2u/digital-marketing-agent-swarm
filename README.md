<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&pause=1000&color=A020F0&center=true&vCenter=true&width=700&lines=Digital+Marketing+Agent+Swarm;23+Agents.+One+Brief.+Zero+API+Keys.;Ollama+%C2%B7+LangGraph+%C2%B7+LangSmith" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-FF6B35?style=for-the-badge)](https://smith.langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-A020F0?style=for-the-badge)](LICENSE)

<br/>

> **One sentence in. Twenty-three specialists out.**  
> A fully local, privacy-first AI swarm that researches, strategises, writes, and reports — no cloud, no cost, no compromise.

</div>

---

```bash
uv run main.py "Launch campaign for a luxury perfume brand in India"
```

```
═════════════════════════════════════════════════════════════════
  MULTI-AGENT PIPELINE  –  Ollama · LangGraph · LangSmith
═════════════════════════════════════════════════════════════════
  Model         : gemma4:latest
  Reviewer Model: llama3.1:8b
  Task          : Launch campaign for a luxury perfume brand in India
═════════════════════════════════════════════════════════════════

[09:17:07][MARKETING DIRECTOR]  Plans marketing research → task validated
[09:18:03][MARKET RESEARCHER]   Analysing market signals...
[09:18:03][TREND ANALYST]       Scanning cultural trends...    ← parallel
[09:42:30][COMPETITOR ANALYST]  Mapping competitor landscape...
[10:06:24][CONTENT STRATEGIST]  Building editorial calendar...
[10:12:08][SEO SPECIALIST]      Mining keyword opportunities...
[10:12:08][PAID ADS]            Structuring paid channels...   ← parallel
[10:12:09][LEAD GENERATION]     Designing lead funnels...      ← parallel
...23 agents later...
[16:04:42][REPORTING]           Executive summary complete ✓

═════════════════════════════════════════════════════════════════
  PIPELINE COMPLETE
═════════════════════════════════════════════════════════════════
  Output files:
    • output/Marketing_Director_Agent_output_20260618.pdf
    • output/Marketing_Research_Agent_output_20260618.pdf
    • output/Competitor_Analyst_Agent_output_20260618.pdf
    • output/SEO_specialist_Agent_output_20260618.pdf
    • ... 23 PDFs total
═════════════════════════════════════════════════════════════════
```

> **Note:** Each agent runs a live DuckDuckGo search, generates a draft, then passes it through a reviewer LLM for up to 2 critique-and-revise rounds before writing its PDF. Expect a full run to take several hours on consumer hardware — this is a thorough research pipeline, not a quick chat.

---

## 🕸️ The Swarm

<div align="center">

```
                      ┌──────────────────────┐
              ┌──────►│  Marketing Director  │◄──────┐
              │       └──────────┬───────────┘       │
              │        ┌─────────┴──────────┐        │
              │  ┌─────▼──────┐    ┌────────▼─────┐  │
              │  │  Market    │    │    Trend     │  │ parallel
              │  │ Researcher │    │   Analyst    │  │ fan-out
              │  └─────┬──────┘    └────────┬─────┘  │
              │        └──────────┬──────────┘        │
              │           ┌───────▼────────┐          │
              │           │   Competitor   │          │
              │           │    Analyst     │          │
              │           └───────┬────────┘          │
              │           ┌───────▼────────┐          │
              │           │    Content     │          │
              │           │  Strategist    │          │
              │           └──┬────┬────┬───┘          │
              │    ┌─────────┘    │    └──────────┐   │
              │ ┌──▼────┐   ┌────▼────┐   ┌──────▼─┐  │
              │ │  SEO  │   │  Paid   │   │  Lead  │  │ parallel
              │ │ Spec. │   │   Ads   │   │  Gen.  │  │ fan-out
              │ └──┬────┘   └────┬────┘   └──────┬─┘  │
              │    │        ┌────▼────┐           │   │
              │    │        │ Google  │           │   │
              │    │        │   Ads   │           │   │
              │    │        └────┬────┘           │   │
              │    │        ┌────▼────┐           │   │
              │    │        │  Meta   │           │   │
              │    │        │   Ads   │           │   │
              │    │        └────┬────┘           │   │
              │    └────────┬────┘────────────────┘   │
              │         ┌───▼──────────┐              │
              │         │   Keyword    │              │
              │         │   Research   │              │
              │         └───┬──────────┘              │
              │     ┌───────┘       └──────────┐      │
              │ ┌───▼──────┐             ┌──────▼───┐ │
              │ │Copywriter│             │ Outreach │ │
              │ └───┬──────┘             └──────┬───┘ │
              │ ┌───▼──────┐             ┌──────▼───┐ │
              │ │ Content  │             │  Email   │ │
              │ │  Writer  │             │ Marketer │ │
              │ └───┬──────┘             └──────┬───┘ │
              │ ┌───▼──────┐                    │     │
              │ │   CRO    │                    │     │
              │ └───┬──────┘                    │     │
              │ ┌───▼──────┐                    │     │
              │ │  Social  │                    │     │
              │ │  Media   │                    │     │
              │ └───┬──┬───┘                    │     │
              │ ┌───▼┐ └──────────────────┐     │     │
              │ │ LI │          ┌──────────▼──┐ │     │
              │ │Spec│          │  Creative   │ │     │
              │ └──┬─┘          │  Director   │ │     │
              │    │            └──────┬──────┘ │     │
              │    │            ┌──────▼──────┐ │     │
              │    │            │Design Agent │ │     │
              │    │            └──────┬──────┘ │     │
              │    │            ┌──────▼──────┐ │     │
              │    │            │  Quality    │ │     │
              │    │            │  Reviewer   │ │     │
              │    │            └──────┬──────┘ │     │
              │    │            ┌──────▼──────┐ │     │
              │    │            │  Analytics  │ │     │
              │    │            └──────┬──────┘ │     │
              │    └──────────┬─────────┘────────┘    │
              │           ┌───▼──────┐                │
              │           │Reporting │                │
              │           └───┬──────┘                │
              └───────────────┘  loops back ──────────┘
```

</div>

---

## 🤖 Agent Roster

| Agent | Specialty | Output |
|---|---|---|
| 🎯 Marketing Director | Orchestration, task gating, loop control | Strategy doc |
| 🔬 Market Researcher | Audience sizing, demand signals | Market analysis |
| 📈 Trend Analyst | Cultural moments, seasonal hooks | Trend report |
| 🕵️ Competitor Analyst | Positioning, pricing, SEO gaps | Intel brief |
| 📋 Content Strategist | Editorial calendar, channel mix | Content plan |
| 🔍 SEO Specialist | Keywords, intent mapping, topic clusters | SEO blueprint |
| 💰 Paid Ads | Channel selection, budget split | Ad strategy |
| 🔴 Google Ads | Search/display campaigns, bidding | Campaign plan |
| 🔵 Meta Ads | Facebook/Instagram creatives, audiences | Ad brief |
| 🗝️ Keyword Research | Primary, secondary, long-tail lists | Keyword map |
| ✍️ Copywriter | Headlines, CTAs, ad copy | Copy deck |
| 📝 Content Writer | Blog posts, landing pages | Long-form content |
| 📊 CRO Specialist | Funnel friction, A/B tests | CRO audit |
| 📱 Social Media Manager | Post calendars, engagement tactics | Social calendar |
| 💼 LinkedIn Specialist | B2B thought leadership | LinkedIn plan |
| 🎨 Creative Director | Visual identity, brand voice | Creative brief |
| 🖌️ Design Agent | Asset specs, style guides | Design brief |
| ✅ Quality Reviewer | Brand compliance, consistency checks | QA report |
| 📉 Analytics | KPIs, dashboards, measurement | Analytics plan |
| 🧲 Lead Generation | Lead magnets, gating, nurture | Lead strategy |
| 📣 Outreach Agent | Influencer, PR, partnership lists | Outreach plan |
| 📧 Email Marketer | Drip sequences, subject lines | Email playbook |
| 📑 Reporting | Executive summary, risks, next steps | Final report |

---

## ⚡ Stack

```python
{
  "primary_llm":   "Ollama  →  gemma4:latest  (fully local, main author)",
  "reviewer_llm":  "Ollama  →  llama3.1:8b   (fully local, critique & approve)",
  "orchestration": "LangGraph StateGraph  (parallel fan-out, supervised loop)",
  "tracing":       "LangSmith",
  "web_search":    "DuckDuckGo  (ddgs)  —  every agent grounds in live data",
  "tool_layer":    "CrewAI BaseTool",
  "pdf_output":    "ReportLab  —  cover page, headings, tables, formatted body",
  "env":           "python-dotenv",
  "package_mgr":   "uv  (recommended)  or  pip",
}
```

---

## 🚀 Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- [Ollama](https://ollama.com/download) installed and running locally

Pull both models before running — the pipeline uses one for authoring and one for reviewing:

```bash
ollama pull gemma4        # primary author model
ollama pull llama3.1:8b   # reviewer / critic model
```

### Install

```bash
# 1. Clone the repository
git clone https://github.com/priyadeep2u/digital-marketing-agent-swarm.git
cd digital-marketing-agent-swarm

# 2. Install dependencies with uv (creates a virtual env automatically)
uv sync

# 3. Alternatively, use pip in a standard venv
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4. Create the output directory
mkdir -p output
```

### Configure

```bash
cp .env.example .env
```

```env
# LangSmith (optional — tracing only; pipeline runs fine without it)
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=digital-marketing-swarm
LANGSMITH_TRACING=true
```

> No OpenAI key. No Anthropic key. No cloud spend. Everything runs on your machine.

### Run

```bash
# With uv (recommended)
uv run main.py "B2B SaaS product launch targeting CTOs in DACH region"

# Or directly if you activated your venv
python main.py "B2B SaaS product launch targeting CTOs in DACH region"

# Built-in demo brief (exclusive perfume brand, India)
uv run main.py
```

Outputs land in `output/` — one PDF per agent plus `graph.png` showing the compiled pipeline.

---

## 📁 Structure

```
digital-marketing-agent-swarm/
│
├── main.py                 # LangGraph pipeline — build_graph() + run()
├── shared.py               # AgentState, LLM config, WriteDocumentTool, ddg_search
│
├── marketing_director.py   # Supervisor / gate
├── market_research.py
├── trend_analyst.py
├── competitor_analyst.py
├── content_strategist.py
├── seo_specialist.py
├── paid_ads.py
├── google_ads.py
├── meta_ads.py
├── keyword_research.py
├── copywriter.py
├── content_writer.py
├── cro_specialist.py
├── social_media.py
├── linkedin_specialist.py
├── creative_director.py
├── design_agent.py
├── quality_reviewer.py
├── analytics.py
├── lead_generation.py
├── outreach_agent.py
├── email_marketer.py
├── reporting.py
│
└── output/                 # 23 PDFs + graph.png generated here
```

---

## 🧠 Architecture Notes

**Deterministic routing** — no LLM decides which node fires next. All routing is hardcoded edges or simple state checks (`state["reporting_decision"] == "REPORTING_COMPLETED"`). Eliminates the infinite loops that plague LLM-routed pipelines.

**Parallel fan-out via LangGraph** — `market_researcher` and `trend_analyst` fire simultaneously. Their outputs fan-in at `competitor_analyst`. The same pattern repeats at `seo_specialist / paid_ads / lead_generation → keyword_research` and `analytics / email_marketer / linkedin_specialist → reporting`.

**Supervised loop** — after `reporting` completes, `marketing_director` re-evaluates `state["reporting_decision"]`. If `REPORTING_COMPLETED`, it routes to `END`. No runaway cycles.

**Dual-LLM review loop** — every agent drafts with `gemma4:latest`, then passes the output to `llama3.1:8b` for structured critique (JSON `{approved, comments}`). The author revises up to 2 rounds before finalising. Reviewer approval short-circuits early.

**Web-grounded agents** — every agent fires a DuckDuckGo search scoped to its specialty before invoking the LLM. Responses reflect live market conditions, not stale training data.

**PDF per agent** — `WriteDocumentTool` (CrewAI `BaseTool` + ReportLab) writes a formatted PDF per agent: cover page, structured headings, bullet lists, rendered Markdown tables, and a footer. Full audit trail of every agent's reasoning.


---

<div align="center">

**Ollama · LangGraph · LangSmith · CrewAI · ReportLab**

*23 agents. Fully local. Drop a ⭐ if it's useful.*

</div>