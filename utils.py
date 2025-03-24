from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio
# test
from dotenv import load_dotenv

load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def llm_call(prompt: str, model: str ="gemini-1.5-flash") -> str:
    """동기 방식 Gemini API 호출"""
    llm = ChatGoogleGenerativeAI(model=model)
    
    message = []
    message.append({"role": "human", "content": prompt})
    response = llm.invoke(message)
    return response.content

async def llm_call_async(prompt: str,model: str ="gemini-1.5-flash") -> str:
    """비동기 방식 Gemini API 호출"""
    llm = ChatGoogleGenerativeAI(model=model)
    
    message = []
    message.append({"role": "human", "content": prompt})
    response = await llm.ainvoke(message)
    return response.content


if __name__ == "__main__":
    test1 = llm_call("Hello, how are you?", "gemini-1.5-flash")
    test2 = asyncio.run(llm_call_async("What is AI", "gemini-1.5-flash"))
    print(test1)
    print(test2)
