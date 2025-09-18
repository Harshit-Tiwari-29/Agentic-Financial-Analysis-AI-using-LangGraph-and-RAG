# Agentic-Financial-Analysis-AI-using-LangGraph-and-RAG

# 🤖 Financial Analysis AI Agent

**Financial Analysis AI Agent** is a full-stack web application featuring a sophisticated AI agent designed for comprehensive financial analysis. Built with **FastAPI** and powered by the **Groq API** for high-speed language model inference, this agent can analyze financial reports, fetch real-time stock data, and perform predictive modeling. The entire application is orchestrated using **LangChain's** agentic frameworks.

The project combines a powerful backend with an intuitive, chat-based frontend to deliver a seamless user experience.

---

## 🌟 Key Features

- **Multi-Tool Agent**: The agent is equipped with three distinct capabilities, and it intelligently chooses the right tool for the job:
  - **RAG on Financial Reports**: Upload a PDF financial report and ask complex questions about its contents.
  - **Real-time Stock Data**: Fetches up-to-the-minute stock prices and key financial metrics using the `yfinance` library.
  - **Price Prediction**: Utilizes a simple machine learning model (Linear Regression) to forecast stock prices for the next 7 days based on historical data.
- **Session-Based Interaction**: Each user interaction is managed in a unique session, ensuring that uploaded documents and conversation history are kept private and isolated.
- **Interactive Web UI**: A clean, modern chat interface built with HTML, Tailwind CSS, and vanilla JavaScript allows for dynamic interaction with the AI agent.
- **High-Speed Inference**: Leverages the Groq LPU™ Inference Engine via `langchain-groq` for extremely fast and responsive agent actions and chat replies.
- **Robust Backend**: Built with FastAPI, providing a scalable and efficient server for handling file uploads, agent queries, and session management.

---

## 🛠️ Installation & Setup

### 1. Prerequisites

- Python 3.8+
- Git

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
