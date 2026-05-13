import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL_ID
from ai.metrics import PrometheusCallbackHandler

async def main():
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL_ID,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        callbacks=[PrometheusCallbackHandler()]
    )
    try:
        await llm.ainvoke([HumanMessage(content="Hello")])
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
