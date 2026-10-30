from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
from typing_extensions import TypedDict
from tavily import TavilyClient
import os

load_dotenv()

# Initialize the Groq model
llm = ChatGroq(
    model="openai/gpt-oss-safeguard-20b",
    temperature=0
)

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


class AgentState(TypedDict):
    question:str
    answer:str
    # status:bool

def chat_boat(state: AgentState):
    query = state["question"]
    system_message="You are a helpful assistant. you give the answer to the question asked by the user. If you need to search the web for the answer, you will say 'I need to search the web for the answer' and then you will search the web for the answer."

    system_message = system_message

    response = llm.invoke(
        query,
        system_message=system_message
    )

    return {
        "answer": [response.content]
    }

def web_search(state:AgentState):
    query=state['question']
    result=tavily_client.search(
        query=query,
        max_result=3,
        search_dept="basic"
    )
    print("Web search result:", result)
    return {"answer": [result]}

def approval_checker(state: AgentState):
    answer = state.get("answer", "")

    if answer is None:
        return "web_search"

    return "chat_boat"



agent = StateGraph(AgentState)

agent.add_node("chat_boat", chat_boat)
agent.add_node("web_search", web_search)
agent.add_node("approval_checker", approval_checker)

agent.add_edge(START, "chat_boat")

agent.add_conditional_edges(
    "approval_checker",
    approval_checker,
    {
        "chat_boat": "chat_boat",
        "web_search": "web_search"
    }
)

agent.add_edge("web_search", END)
agent.add_edge("chat_boat", END)

app=agent.compile()

# 5. Pass user input into the question key
user_input = "What was the result of the latest India vs Australia match?"

result = app.invoke({
    "question": user_input
})

print("LLM answer:", result["answer"])

