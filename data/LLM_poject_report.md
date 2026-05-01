# **Autonomous Equity Research Agent: A Multi-Agent System for Institutional-Grade Financial Analysis**

**Executive Summary**
[cite_start]This report presents the design, implementation, and evaluation of an autonomous equity research agent capable of generating institutional-quality financial research reports[cite: 5, 6]. [cite_start]The system addresses a critical inefficiency in the sell-side research industry: traditional equity research reports require **2-3** weeks of analyst effort, yet follow a highly standardized eight-section structure[cite: 7]. [cite_start]Our solution combines retrieval-augmented generation (RAG), bank-specific financial modeling, and agentic orchestration via LangGraph to produce comprehensive research reports in under five minutes at **$0.22** per report[cite: 8]. [cite_start]The system demonstrates **71%** accuracy on key financial metrics when validated against real-time market data, while achieving an **84%** cost reduction compared to GPT-4o through strategic use of Gemini 2.5 Flash[cite: 9]. [cite_start]Key innovations include a pre-generation pattern to eliminate table corruption, a bank-specific Free Cash Flow to Equity (FCFE) model accounting for Basel III requirements, and a multi-source RAG pipeline[cite: 10].

***

**1. Problem Statement**
[cite_start]Fundamental challenges prevent the naive application of large language models to this domain[cite: 16]:
* [cite_start]**Numerical hallucination**: LLMs frequently generate plausible but incorrect financial figures, rendering outputs unsuitable for decision-making contexts where precision is paramount[cite: 17].
* [cite_start]**Structural corruption**: Markdown tables embedded in JSON response strings systematically corrupt during parsing, breaking the tabular financial statements central to equity research[cite: 18].

***

**2. System Architecture and Design**

**2.1 LangGraph Orchestration Framework**
[cite_start]The system implements a stateful agent using LangGraph, a framework for building directed acyclic graphs (DAGs) where each node is a Python function and edges represent conditional transitions[cite: 23, 35]. [cite_start]This architecture provides explicit control flow superior to naive ReAct-style prompting, which often produces brittle loops in complex multi-step tasks[cite: 36].

**2.2 State Schema and Nodes**
[cite_start]The graph state is a typed dictionary with seven fields: `ticker`, `section`, `messages`, `sections_done`, `tool_calls` (capped at **12**), and `chart_path`[cite: 38, 39]. The six primary nodes are:
* [cite_start]**plan**: Initializes section-specific context by injecting targeted system prompts[cite: 43].
* [cite_start]**llm**: Invokes Gemini 2.5 Flash via OpenRouter API[cite: 45].
* [cite_start]**parse**: Extracts and validates structured JSON from LLM responses[cite: 47].
* [cite_start]**tool**: Executes requested data retrieval or deterministic computation[cite: 48].
* [cite_start]**save**: Stores completed section markdown and advances the state[cite: 49].
* [cite_start]**compile**: Assembles the final 8-section report into a canonical order[cite: 50].

[cite_start]Two table-heavy sections (financial performance and valuation) are pre-generated entirely in Python[cite: 67, 70]. [cite_start]By never allowing the LLM to generate these numbers or tables, the system achieves zero numerical hallucination and guarantees structural validity[cite: 21, 22].

**2.3 Retrieval-Augmented Generation Pipeline**
[cite_start]The RAG system integrates three distinct data sources[cite: 103]:
* [cite_start]**10-K Reports**: ~**320** chunks using a **200**-word sliding window for business segments and geography[cite: 105, 107].
* [cite_start]**Earnings Call Transcripts**: ~**90** chunks per quarter for management commentary and margin trends[cite: 108, 109].
* [cite_start]**Regulatory Filings (FR Y-9C, FFIEC 102)**: ~**78** chunks covering **5** years of market risk data like VaR and RWA[cite: 110].

[cite_start]Retrieval uses FAISS cosine similarity for top-**5** chunks[cite: 111, 115]. [cite_start]Live market data (price, P/E, P/B, ROE) is fetched from Yahoo Finance via `yfinance` with defensive field lookups[cite: 125, 126, 133].

***

**3. Financial Modeling Engine**
* [cite_start]**Bank-Specific FCFE Model**: Bank valuation uses Free Cash Flow to Equity (FCFE) because interest expense is a core revenue item[cite: 136, 137]. [cite_start]Capital expenditure is replaced by regulatory capital (CET1) reinvestment required under **Basel III**[cite: 138, 141].
* [cite_start]**Valuation Results**: For Goldman Sachs (GS), the system produced a blended price target of **$531.12** (HOLD), weighted equally across FCFE DCF (**$517.33**), P/B Multiple (**$435.40**), and Gordon Growth Implied P/B[cite: 162, 163, 175].
* [cite_start]**Assumptions**: The cost of equity is **14%** (derived from CAPM with a **4.2%** risk-free rate and **1.45** beta)[cite: 150, 152, 153]. [cite_start]Terminal value is calculated with a conservative **1%** growth rate[cite: 157, 158].

***

**4. Results**
* [cite_start]**Accuracy**: Validation shows **71%** accuracy (5 of 7 metrics within a 5% tolerance)[cite: 9]. [cite_start]Discrepancies, such as the **10%** beta variance, are largely attributable to yfinance data lag compared to premium sources like Bloomberg[cite: 153].
* [cite_start]**Cost Efficiency**: Gemini 2.5 Flash costs **$0.22** per report compared to **$8.07** for GPT-4o, representing a **97%** cost reduction[cite: 32].

***

**5. Limitations**
* [cite_start]**Data Mismatch**: The Yahoo Finance schema assumes deposit-funded banks, requiring estimations for investment banks like GS that have minimal deposits[cite: 133].
* [cite_start]**Beta Variance**: A **10%** gap exists between yfinance (**1.45**) and Bloomberg (**1.31**) due to different lookback windows[cite: 153].
* **Industry Analysis**: Currently, forward-looking TAM figures rely on LLM training data rather than real-time retrieved documents.

***

**6. Conclusion**
[cite_start]The project successfully automated institutional-quality equity research by separating LLM contextual synthesis from deterministic Python computation[cite: 21]. [cite_start]This architecture achieves zero numerical hallucination and generates **3,500**-word reports in under **5** minutes[cite: 8]. [cite_start]At a **$0.22** price point, the system enables daily coverage for over **500** stocks and real-time analysis for retail platforms[cite: 8, 32]. [cite_start]Future work will prioritize wrapping the system in a FastAPI endpoint on AWS Lambda to support **10,000** reports per month for less than **$100**[cite: 101].

***

**References**
1.  [cite_start]**Regulatory Filings**: FR Y-9C Consolidated Financial Statements and FFIEC 102 Market Risk Regulatory Report[cite: 110].
2.  [cite_start]**Corporate Reports**: Goldman Sachs Group, Inc. 2024 Annual Report (Form 10-K)[cite: 105].
3.  **Similarity Search**: Johnson, J. et al. (2019). [cite_start]"Billion-scale similarity search with GPUs"[cite: 111].
4.  [cite_start]**Market Data**: Yahoo Finance (2026) GS Stock Data and yfinance Python Library documentation[cite: 125, 133].
