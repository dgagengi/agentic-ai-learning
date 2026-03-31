# Create an AI agent using CrewAI, Groq, SerpAPI and Yahoo Finance. 

# Step 1: Install necessary libraries
import os
import streamlit as st
from crewai import Agent, Crew, Task, LLM
from crewai.tools import BaseTool
import yfinance as yf
from serpapi import GoogleSearch
from typing import Type
from pydantic import BaseModel, Field
import warnings 

warnings.filterwarnings("ignore")

st. set_page_config(page_title="Stock Analysis Agent", page_icon=":guardsman:", layout="wide")
st.title("AI Stock Analysis Agent :guardsman:")
st. markdown("## This agent uses CrewAI, Groq, SerpAPI, and Yahoo Finance to analyze stock data and provide insights.")

## Side bar for the API keys
with st.sidebar:
    st.header("API Keys")
    groq_api_key = st.text_input("Groq API Key", type="password")
    st.markdown("Get your Groq API key from [Groq's website](https://www.groq.com/).")
   
    serpapi_api_key = st.text_input("SerpAPI Key", type="password")
    st.markdown("Get your SerpAPI key from [SerpAPI's website](https://serpapi.com/).")
    st.markdown("### Information: ###")
    st.info("Groq and SerpAPI are free and safe to use.")

## Check if the keys are provided
if not groq_api_key or not serpapi_api_key:
    st.warning("Please provide both Groq and SerpAPI API keys to proceed.")
    st.stop()

## Set Environment Variables
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["SERPAPI_API_KEY"] = serpapi_api_key

# Initialize LLM
@st.cache_resource
def get_llm(groq_key):
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.7
    )

llm = get_llm(groq_key)

# Define Tools
class StockSearchInput(BaseModel):
    query: str = Field(..., description="Search query")

class YahooFinanceInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker")

class StockSearchTool(BaseTool):
    name: str = "StockNewsSearcher"
    description: str = "Search for latest stock news using SerpAPI Google News"
    args_schema: Type[BaseModel] = StockSearchInput

    def _run(self, query: str) -> str:
        """Search for stock news using SerpAPI"""
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": os.environ.get("SERPAPI_API_KEY"),
                "tbm": "nws",
                "num": 3
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            news = results.get("news_results", results.get("organic_results", []))

            if not news:
                return "No recent news found"
            output = []
            for item in news[:3]:
                title = item.get("title", "")
                snippet = item.get("snippet", item.get("description", ""))
                output.append(f"• {title}: {snippet[:100]}")

            return "\n".join(output)

        except Exception as e:
            return f"Error: {str(e)}"

class YahooFinanceTool(BaseTool):
    name: str = "YahooFinanceFetcher"
    description: str = "Get stock price data from Yahoo Finance"
    args_schema: Type[BaseModel] = YahooFinanceInput

    def _run(self, ticker: str) -> str:
        """Fetch stock price data"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")

            if hist.empty:
                return f"No data for {ticker}"

            latest = hist.tail(5)
            current = latest['Close'].iloc[-1]
            change = ((latest['Close'].iloc[-1] - latest['Close'].iloc[0]) / latest['Close'].iloc[0]) * 100

            return f"""Stock: {ticker}\nPrice: ${current:.2f}\n5-day Change: {change:+.2f}%\nHigh: ${latest['High'].max():.2f}\nLow: ${latest['Low'].min():.2f}"""

        except Exception as e:
            return f"Error: {str(e)}"
        
# Initialize Tools
stock_search_tool = StockSearchTool()
yahoo_finance_tool = YahooFinanceTool()

# Create Agents
@st.cache_resource
def get_agents(_llm):
    analyst = Agent(
        role='Stock Analyst',
        goal='Analyze stocks',
        backstory='Financial expert',
        verbose=False,
        llm=_llm,
        tools=[stock_search_tool, yahoo_finance_tool]
    )

    writer = Agent(
        role='Report Writer',
        goal='Write reports',
        backstory='Financial writer',
        verbose=False,
        llm=_llm
    )

    return analyst, writer

analyst_agent, writer_agent = get_agents(llm)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input("Enter Stock Ticker", value="AAPL", max_chars=10).upper()

with col2:
    analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

if analyze_btn:
    if not ticker:
        st.error("Please enter a stock ticker")
    else:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Create tasks
                news_task = Task(
                    description=f"Search latest news for {ticker} stock",
                    expected_output="News summary",
                    agent=analyst_agent
                )

                price_task = Task(
                    description=f"Analyze {ticker} price trends",
                    expected_output="Price analysis",
                    agent=analyst_agent
                )

                report_task = Task(
                    description=f"""Write investment report for {ticker}.
                Include: Summary, News, Price Analysis, Outlook.
                Keep under 300 words.""",
                    expected_output="Investment report",
                    agent=writer_agent
                )
                # Create and run crew
                crew = Crew(
                    agents=[analyst_agent, writer_agent],
                    tasks=[news_task, price_task, report_task],
                    verbose=False
                )

                result = crew.kickoff()

                # Convert result to string and escape dollar signs
                result_text = str(result).replace('$', r'\$')

                # Display results
                st.success("Analysis Complete!")

                st.markdown("---")
                st.markdown(f"## 📊 Investment Report - {ticker}")
                st.markdown(result_text)
                st.markdown("---")

                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=str(result),
                    file_name=f"{ticker}_analysis.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Powered by [CrewAI](https://crewai.com) and [Groq](https://groq.com)")
