start ollama

use this 
Python 3.10.13
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

curl -X POST "http://localhost:8000/chat" \            
  -H "Content-Type: application/json" \
  -d '{"sessionid": null, "message": "What is the interest rate of gold loan in rapipay?"}'