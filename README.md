# Agentic-Financial-Analysis-AI-using-LangGraph-and-RAG

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
```
3. Set Up a Virtual Environment
```Bash

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```
4. Install Dependencies
The required packages are listed in requirements.txt. Install them using pip:

```Bash

pip install -r requirements.txt
```
5. Configure Environment Variables
Sign up at Groq to get your free API key.

Create a .env file in the project's root directory.

Add your API key to the .env file like this:

Code snippet

GROQ_API_KEY="your_groq_api_key_here"
🚀 Usage Guide
To launch the web application, run the main.py file using Uvicorn:

```Bash
uvicorn main:app --reload
```
The application will be available at https://www.google.com/search?q=http://127.0.0.1:8000.

Open this URL in your browser to access the chat interface.

Step 1: Upload a financial report in PDF format. A system message will confirm when it's processed.

Step 2: Ask questions. You can query the document, fetch stock data, or ask for a price prediction.

Example Prompts:

After uploading a report: "What were the total revenues mentioned in the document?"

For stock data: "What is the current price and market cap for AAPL?"

For prediction: "Predict the stock price for NVDA"

⚙️ Technical Details
```Core Libraries
FastAPI: Serves as the web framework for the backend API.

LangChain: The core framework used to build the ReAct agent, manage tools, and orchestrate the LLM workflow.

ChatGroq: Provides the LangChain integration with the Groq API, using the llama3-8b-8192 model.

FAISS (Facebook AI Similarity Search): An in-memory vector store used for the RAG functionality.

Sentence-Transformers: Generates the embeddings for the text chunks from the uploaded PDF.

yfinance: Fetches live stock market data.

scikit-learn: Used to build and train the simple linear regression model for price prediction.
```
Application Architecture
Frontend-Backend Separation: The UI is built with standard index.html, style.css, and script.js files, which communicate with a completely separate FastAPI backend.

Session Management: A simple in-memory Python dictionary on the backend tracks each user session via a unique session_id. Each session contains its own vector database, agent executor, and conversation memory.

Agentic Workflow (ReAct): The application uses a ReAct (Reasoning and Acting) agent. The agent receives a prompt, thinks about which tool to use, executes the tool with the necessary input, observes the result, and repeats this loop until it can provide a final answer.

RAG Pipeline: When a PDF is uploaded, it is read, split into text chunks, converted into vector embeddings, and stored in a session-specific FAISS vector store for fast retrieval.

💡 Prompt Design
The agent's behavior is heavily guided by a custom prompt template. This template instructs the agent on how to reason, use tools, and format its output. A critical part of the prompt ensures the agent is proactive in asking for clarification.

Agent Prompt Template
Python

PROMPT_TEMPLATE = """
You are a helpful financial assistant. Respond to the user's questions as best as you can.

You have access to the following tools:

{tools}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action


When you have a response to say to the user, or if you need to ask for more information, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response or question to the user]


**CRITICAL RULE:** If you need more information from the user to use a tool (e.g., a stock ticker symbol), you MUST stop and ask the user for it by using the "Final Answer" format. Do not try to guess or make up information.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""
🧠 Challenges & Solutions
Challenge 1: Integrating Multiple, Distinct Tools
Problem: The agent needs to reliably choose the correct tool (RAG, stock fetcher, or predictor) based on the user's query.
Solution: The ReAct prompt template was designed with clear, descriptive tool names and descriptions. This gives the LLM the necessary context to make an informed decision on which tool to use.

Challenge 2: Managing User-Specific Data (RAG)
Problem: Each user uploads a different financial report. The RAG system must only query the document relevant to the current user.
Solution: A session-based architecture was implemented. When a user connects, they get a unique session_id. The FAISS vector store is created and stored in a dictionary keyed by this session_id, ensuring complete data isolation between users.

Challenge 3: Handling Ambiguous User Queries
Problem: A user might ask, "predict the price" without specifying a stock ticker. The agent could fail or hallucinate a ticker.
Solution: The prompt includes a "CRITICAL RULE" that explicitly instructs the agent to stop and ask the user for missing information before using a tool. This makes the agent more robust and interactive.

Challenge 4: Presenting Structured Tool Output Cleanly
Problem: The stock data and prediction tools return data in a JSON format, which is not user-friendly if displayed as a raw string.
Solution: The frontend JavaScript includes a formatAnswer function that detects JSON-like strings in the agent's response. It then parses this string and renders it as a neatly formatted HTML block, improving readability.

📂 File Structure Overview
```project-root/
│
├── main.py                # The main FastAPI application logic.
├── tools.py               # Defines the tools available to the LangChain agent.
├── ml_model.py            # Contains the stock price prediction model logic.
├── requirements.txt       # Lists all Python dependencies for the project.
├── .env                   # Stores environment variables (e.g., GROQ_API_KEY).
│
├── index.html             # The main HTML file for the user interface.
└── static/
    ├── script.js          # JavaScript for frontend logic (API calls, DOM manipulation).
    ├── styles.css         # CSS for styling the chat interface.
    └── favicon.ico        # The application's favicon.
```
📄 License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it for personal or professional use.

🙌 Acknowledgments
Groq for providing the high-speed LPU™ Inference Engine.

LangChain for the powerful agent and tool management framework.

FastAPI for the excellent web framework.

