from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tools import rapipay_loan_tool, weather_tool, cricket_tool, news_tool, flights_tool, fallback_tool

app = FastAPI()

class QueryModel(BaseModel):
    query: str

class CityModel(BaseModel):
    city: str

class CricketModel(BaseModel):
    team1: str
    team2: str

class NewsModel(BaseModel):
    country: str

class FlightsModel(BaseModel):
    source: str
    destination: str

@app.post("/tools/rapipay_loan_tool")
def call_rapipay_loan_tool(data: QueryModel):
    return {"result": rapipay_loan_tool(data.query)}

@app.post("/tools/weather_tool")
def call_weather_tool(data: CityModel):
    return {"result": weather_tool(data.city)}

@app.post("/tools/cricket_tool")
def call_cricket_tool(data: CricketModel):
    return {"result": cricket_tool(data.team1, data.team2)}

@app.post("/tools/news_tool")
def call_news_tool(data: NewsModel):
    return {"result": news_tool(data.country)}

@app.post("/tools/flights_tool")
def call_flights_tool(data: FlightsModel):
    return {"result": flights_tool(data.source, data.destination)}

@app.post("/tools/fallback_tool")
def call_fallback_tool(data: QueryModel):
    return {"result": fallback_tool(data.query)}

# Add this at the end of the file to run with: python mcp_server.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=9000, reload=True)
