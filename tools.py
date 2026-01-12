# tools.py
import os
import json
import yfinance as yf
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from ml_model import predict_stock_price

def get_rag_tool(db):
    """Creates a tool for querying the financial report vector database."""
    if db is None:
        # Return a tool that informs the user to upload a document first
        return Tool(
            name="Financial Report QA System",
            func=lambda q: "Please upload a financial report first before asking questions about it.",
            description="Useful for when you need to answer questions about a financial report. You must upload a report first."
        )
        
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192", temperature=0)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    
    return Tool(
        name="Financial Report QA System",
        func=qa_chain.run,
        description="Useful for when you need to answer questions about the uploaded financial report. Input should be a clear, natural language question."
    )

def get_stock_data_tool():
    """Creates a tool for fetching real-time stock data."""
    def get_data(ticker: str) -> str:
        """Fetches stock data and returns it as a JSON string."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            # Check if essential data is available
            if not info or info.get('regularMarketPrice') is None:
                 return f"Could not find valid stock data for ticker '{ticker}'. It may be an invalid symbol."

            data = {
                "company_name": info.get("longName"),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": f"${info.get('marketCap', 0):,}",
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "previous_close": info.get("previousClose"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": f"{info.get('dividendYield', 0) * 100:.2f}%"
            }
            # Convert the dictionary to a JSON string for the agent
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error fetching data for ticker {ticker}: {e}. Please ensure the ticker is correct."

    return Tool(
        name="Stock Data Fetcher",
        func=get_data,
        description="Useful for fetching real-time stock price and key financial metrics for a given stock ticker. Input should be the stock ticker symbol (e.g., 'AAPL', 'GOOGL')."
    )

def get_prediction_tool():
    """Creates a tool for predicting stock prices."""
    def run_prediction(ticker: str) -> str:
        """Runs the prediction model and returns the result as a JSON string."""
        result = predict_stock_price(ticker)
        # If the result is a dictionary, convert it to a JSON string
        if isinstance(result, dict):
            return json.dumps(result, indent=2)
        # Otherwise, it's likely an error message string, so return as is
        return str(result)

    return Tool(
        name="Stock Price Predictor",
        func=run_prediction,
        description="Useful for predicting the stock price for the next 7 days. Input should be the stock ticker symbol (e.g., 'TSLA', 'MSFT')."
    )
