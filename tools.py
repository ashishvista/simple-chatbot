from langchain_core.tools import tool

def get_vectordb():
    # Import here to avoid circular import
    from .main import vectordb
    return vectordb

@tool
def rapipay_loan_tool(query: str) -> str:
    """Useful for questions about loan interest rates, EMIs, or loan products from Rapipay"""
    vectordb = get_vectordb()
    try:
        docs = vectordb.similarity_search(query, k=3)
        result = ["Here's what I found about Rapipay loans:"]
        for i, doc in enumerate(docs, 1):
            result.append(f"\nDocument {i}:\n{doc.page_content}\n")
        return "\n".join(result) if len(result) > 1 else "No information found about Rapipay loans"
    except Exception as e:
        return f"Error searching loan info: {str(e)}"

@tool
def weather_tool(city: str) -> str:
    """Get weather info for a city.
    
    Args:
        city (str): The name of the city to get weather information for.
    """
    if not city:
        raise ValueError("City is required for weather info.")
    return str({"location": city, "forecast": "Sunny", "temperature": "35C"})

@tool
def cricket_tool(team1: str, team2: str) -> str:
    """Get cricket match scores."""
    if not team1 or not team2:
        raise ValueError("Both team names are required for cricket score.")
    return str({"match": f"{team1} vs {team2}", "score": "250/3", "status": f"{team1} batting"})

@tool
def news_tool(country: str) -> str:
    """Get top 10 news for a country."""
    if not country:
        raise ValueError("Country is required for news.")
    return str({"country": country, "top_10_news": [f"News {i}" for i in range(1, 11)]})

@tool
def flights_tool(source: str, destination: str) -> str:
    """Get flight details between two cities."""
    if not source or not destination:
        raise ValueError("Source and destination cities are required for flight details.")
    return str({"flight": "AI202", "status": "On Time", "departure": source, "arrival": destination})

tools = [rapipay_loan_tool, weather_tool, cricket_tool, news_tool, flights_tool]
