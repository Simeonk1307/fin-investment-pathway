# Real-Time Financial Intelligence System

**A streaming-native agentic AI platform for financial analysis using Pathway, Multi-Agent LLMs, and Knowledge Graphs**

Inter IIT Tech Meet 14.0 - Pathway Problem Statement

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Running the System](#running-the-system)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

---

## Problem Statement

Retail investors lack access to real-time financial intelligence. Our system provides:
- **Sub-5 second latency** from news event to investment signal
- **Multi-source data fusion**: Live stock prices, news, SEC filings, social sentiment
- **AI-powered analysis**: Multi-agent collaboration with guardrails
- **Streaming-native**: Pathway-powered incremental computation

---

## Quick Start

### Prerequisites

- **Python**: 3.12 or higher
- **Disk Space**: 11GB (for dependencies)
- **Redpanda Cloud**: Free tier account ([sign up](https://redpanda.com/try-redpanda))
- **API Keys**:
  - FinnHub (required) - [Get free key](https://finnhub.io/)
  - OpenAI (required) - [Get key](https://platform.openai.com/)
  - Groq (optional) - [Get key](https://console.groq.com/)

### Installation

```bash
# 1. Clone repository
cd fin-investment-pathway

# 2. Create Python virtual environment
python3 -m venv psvenv
source psvenv/bin/activate  # Windows: psvenv\Scripts\activate

# 3. Install dependencies
pip install -r freezed-requirements.txt
```

### Configuration

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env file with your credentials
nano .env  # or use any text editor
```

**Required environment variables:**

```ini
# Pathway License (trial key available)
PATHWAY_LICENSE_KEY=your-pathway-license-key

# Redpanda Cloud credentials
REDPANDA_BROKERS=your-cluster.redpanda.com:9092
REDPANDA_USERNAME=your-username
REDPANDA_PASSWORD=your-password
REDPANDA_SECURITY_PROTOCOL=SASL_SSL
REDPANDA_SASL_MECHANISM=SCRAM-SHA-256

# API Keys
FINNHUB_API_KEY=your-finnhub-key
OPENAI_API_KEY=your-openai-key

# Optional
GROQ_API_KEY=your-groq-key

# Tickers to track (adjust as needed)
TICKERS=["RELIANCE.NS","HDFCBANK.NS","TCS.NS","INFY.NS","ICICIBANK.NS"]
```

---

##  System Architecture

```
External Data Sources
    ↓
┌─────────────────────────────────┐
│  Bronze Layer (Raw Ingestion)  │
│  • FinnHub Stocks (WebSocket)   │
│  • FinnHub News (REST)          │
│  • FinnHub Filings (REST)       │
│  • Reddit HTML Scraper          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Redpanda Topics (Kafka API)   │
│   bronze.stocks, bronze.news    │
│   bronze.filings, bronze.socials│
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Silver Layer (Validation)      │
│  • Schema Validation            │
│  • Deduplication                │
│  • Dead Letter Queues (DLQ)     │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Gold Layer (Analytics)         │
│  • Technical Indicators (SMA/RSI)│
│  • Sentiment Analysis (FinBERT) │
└─────────────────────────────────┘
    ↓
┌────────────────┬────────────────┬──────────────────┐
│  Knowledge     │  LSTM Price   │ Multi-Agent      │
│  Graph (Neo4j) │  Forecasting  │  System          │
│  • Entities    │  • Shadow     │  • News Analyst  │
│  • Relations   │    Training   │  • Filings       │
│  • Events      │  • 6h Cycle   │  • Social        │
│                │               │  • Synthesizer   │
│                │               │  • Guardrails    │
└────────────────┴────────────────┴──────────────────┘
    ↓
Outputs: trading_signals.jsonl, news_analysis.csv, predictions.jsonl
```

---

##  Running the System

The system runs as separate pipeline components. Open multiple terminal windows:

### Terminal 1: Main Pipeline

```bash
source psvenv/bin/activate
PYTHONPATH=. python src/main_pipeline.py
```

This starts the core Bronze → Silver → Gold data flow.

### Terminal 2: Input Pipeline (Optional - for testing)

```bash
source psvenv/bin/activate  
PYTHONPATH=. python src/input_pipeline.py
```

Injects historical data for testing outside market hours.

### Check Outputs

Monitor the `outputs/` directory for generated files:

```bash
# View latest signals
tail -f outputs/trading_signals.jsonl

# View agent recommendations
cat outputs/news_analysis.csv
```

---

##  Project Structure

```
fin-investment-pathway/
├── src/
│   ├── layers/
│   │   ├── bronze_layer/        # Data collectors (FinnHub, Reddit)
│   │   ├── silver_layer/        # Validation, deduplication
│   │   └── gold_layer/          # Analytics (indicators, sentiment)
│   ├── agents/                  # Multi-agent system (LangGraph)
│   ├── KnowledgeGraph/          # Neo4j integration
│   ├── schemas/                 # Pydantic data contracts
│   ├── utils/                   # Shared utilities
│   ├── config/                  # Configuration management
│   ├── input_pipeline.py        # Historical data injector
│   └── main_pipeline.py         # Main orchestrator
├── config/                      # External configs
├── outputs/                     # Pipeline outputs
├── freezed-requirements.txt     # Python dependencies
├── .env.example                 # Environment template
├── architecture.mmd             # Mermaid architecture diagram
├── BENCHMARKS.md                # Performance metrics
├── README.md                    # This file
└── LICENSE                      # MIT License
```

---

## Results

### Performance Metrics

| Metric | Value | Baseline |
|--------|-------|----------|
| End-to-end latency (p50) | 1.9s | 45s (batch) |
| End-to-end latency (p95) | 4.7s | - |
| Deduplication accuracy | 99.8% | 92% |
| Agent F1 score | 0.86 | 0.62 (single LLM) |
| LSTM MAE | 1.8% | 3.2% (ARIMA) |
| Throughput | 850 msg/s | - |

### Backtesting (Oct-Nov 2024)

- **Portfolio Return**: +10.95% (vs Buy-Hold -0.51%)
- **Sharpe Ratio**: 1.42 (vs 0.18)
- **Max Drawdown**: -4.2% (vs -12.7%)
- **Win Rate**: 62% (24/39 trading days)

### Cost

- **Total**: $429/month ($0.018 per signal)
- **vs Bloomberg Terminal**: $2,000/month (5x cheaper)

---

##  Key Features

1. **Streaming-Native Architecture**
   - Pathway incremental computation (O(1) updates)
   - Window-based aggregations (5m, 20m, 50m)
   - Live Knowledge Graph updates

2. **Multi-Agent Collaboration**
   - News Analyst (sentiment + entity extraction)
   - Filings Analyst (RAG over SEC filings)
   - Social Analyst (Reddit aggregation)
   - Final Synthesizer (GPT-4)
   - Guardrail Agent (hallucination filtering)

3. **Robust Error Handling**
   - Exponential backoff (5s → 20s)
   - Dead Letter Queues for invalid data
   - Timeout escalation (10s → 50s)

4. **Adaptive LSTM**
   - Shadow training every 6 hours
   - Automatic model swapping on improvement
   - 44% better accuracy vs ARIMA

---

##  Infrastructure

- **Streaming**: Redpanda Cloud (3-node cluster, AWS ap-south-1)
- **Knowledge Graph**: Neo4j (100K nodes)
- **OTEL Stack**: Loki (logs) + Prometheus (metrics) + Grafana (dashboards)
- **Compute**: AWS EC2 t3.xlarge (4 vCPU, 16GB RAM)

---

##  Known Limitations

1. **Market Hours**: Live stock data only works during NSE hours (9:15 AM - 3:30 PM IST)
2. **API Rate Limits**:
   - FinnHub: 60 calls/min (free tier)
   - OpenAI: 10K tokens/min (paid tier required)
3. **Scalability**: Max 50 concurrent tickers (CPU limited)

---

##  License

This project uses MIT License. See `LICENSE` file for details.

Third-party dependencies:
- Pathway: Commercial license (trial key)
- LangChain: MIT
- HuggingFace: Apache 2.0
- Redpanda: Business Source License 1.1

---

##  Acknowledgments

- **Pathway** - Streaming computation framework
- **Redpanda** - Kafka-compatible platform
- **OpenAI** - GPT-4 API access
- **HuggingFace** - FinBERT model
- **Inter IIT Organizers** - Problem statement

---

##  Contact

For questions during evaluation:
- Check `/outputs` folder for sample outputs
- Run `src/input_pipeline.py` to simulate live data
- See `BENCHMARKS.md` for detailed metrics

**Team**: Team 28   
**Competition**: Inter IIT Tech Meet 14.0

---

##  Citation

```bibtex
@misc{fin-investment-pathway-2024,
  title={Real-Time Financial Intelligence: A Streaming-Native Agentic Platform},
  author={Team 28},
  year={2024},
  howpublished={Inter IIT Tech Meet 14.0 - Pathway Agentic AI}
}
```
