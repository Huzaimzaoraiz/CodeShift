import os
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="deepseek-r1-distill-llama-70b", # Assuming this is what they use
    temperature=0.1,
    max_tokens=3000,
    max_retries=1
)

try:
    print("Invoking LLM...")
    res = llm.invoke("Hello, think step by step.")
    print("Response received.")
    print(res.content[:200])
except Exception as e:
    print(f"Error: {e}")
