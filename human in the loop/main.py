import os

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from tavily import TavilyClient

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


# =========================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from your .env file")


# =========================================================
# STEP 2: INITIALIZE LLM AND TAVILY
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# =========================================================
# STEP 3: DEFINE AGENT STATE
# =========================================================

class AgentState(TypedDict, total=False):
    question: str

    needs_web_search: bool
    search_reason: str

    approved: bool

    answer: str


# =========================================================
# STEP 4: CLASSIFY THE USER'S QUESTION
# =========================================================

def classify_question(state: AgentState):

    question = state["question"]

    system_prompt = """
    You are a question-routing assistant.

    Decide whether answering the user's question requires a web search.

    Return exactly one of these two words:

    WEB
    DIRECT

    Choose WEB when the question requires current or changing information,
    such as:
    - Latest sports results
    - Current news
    - Today's weather
    - Current prices
    - Recent events
    - Information that may have changed recently

    Choose DIRECT when the question can reasonably be answered using
    general knowledge, such as:
    - Python concepts
    - Programming explanations
    - Basic mathematics
    - General educational questions
    - Stable technical concepts

    Do not perform a web search yourself.
    Return only WEB or DIRECT.
    """

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
    )

    decision = response.content.strip().upper()

    print("\n--- QUESTION CLASSIFICATION ---")
    print("Question:", question)
    print("Model decision:", decision)

    # Normalize the model's response.
    if decision.startswith("WEB"):
        needs_search = True
        reason = "The question may require current information."
    else:
        needs_search = False
        reason = "The question can be answered using general knowledge."

    return {
        "needs_web_search": needs_search,
        "search_reason": reason
    }


# =========================================================
# STEP 5: ROUTE AFTER CLASSIFICATION
# =========================================================

def route_after_classification(state: AgentState):

    if state["needs_web_search"]:
        return "approval_checker"

    return "chat_boat"


# =========================================================
# STEP 6: ASK THE USER FOR WEB SEARCH PERMISSION
# =========================================================

def approval_checker(state: AgentState):

    # interrupt() pauses the graph until the user responds.
    user_decision = interrupt(
        {
            "message": "This question may require a web search.",
            "question": state["question"],
            "reason": state["search_reason"],
            "options": [
                "yes",
                "no"
            ]
        }
    )

    # This code runs when the graph resumes after the user responds.
    approved = (
        user_decision is True
        or (
            isinstance(user_decision, str)
            and user_decision.strip().lower() in ("yes", "y")
        )
    )

    print("\n--- HUMAN APPROVAL ---")
    print("Web search approved:", approved)

    return {
        "approved": approved
    }


# =========================================================
# STEP 7: ROUTE AFTER USER APPROVAL
# =========================================================

def route_after_approval(state: AgentState):

    if state["approved"]:
        return "web_search"

    # If the user rejects the search, answer directly.
    return "chat_boat"


# =========================================================
# STEP 8: ANSWER WITHOUT WEB SEARCH
# =========================================================

def chat_boat(state: AgentState):

    question = state["question"]

    system_prompt = """
    You are a helpful AI assistant.

    Answer the user's question using your existing knowledge.

    Do not perform or claim to have performed a web search.

    If the question requires current information that you cannot verify,
    clearly explain that your answer may not be up to date.

    Give a clear, useful answer.
    """

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
    )

    print("\n--- DIRECT ANSWER GENERATED ---")

    return {
        "answer": response.content
    }


# =========================================================
# STEP 9: SEARCH THE WEB USING TAVILY
# =========================================================

def web_search(state: AgentState):

    question = state["question"]

    print("\n--- SEARCHING THE WEB ---")
    print("Search query:", question)

    search_result = tavily_client.search(
        query=question,
        max_results=3,
        search_depth="basic"
    )

    results = search_result.get("results", [])

    if not results:
        return {
            "answer": (
                "I could not find useful web search results for "
                "your question. Please try a more specific query."
            )
        }

    # Convert the search results into readable text.
    context_parts = []

    for index, item in enumerate(results, start=1):

        title = item.get("title", "No title")
        url = item.get("url", "No URL")
        content = item.get("content", "No content available")

        context_parts.append(
            f"""
            Result {index}
            Title: {title}
            URL: {url}
            Content: {content}
            """
        )

    context = "\n\n".join(context_parts)

    print("\n--- WEB RESULTS RETRIEVED ---")
    print(context)

    # Ask the LLM to answer using the retrieved information.
    system_prompt = """
    You are an AI assistant answering a question using web search results.

    Instructions:
    1. Use the supplied search results to answer the question.
    2. Do not invent facts that are not supported by the results.
    3. If the results are insufficient, explain the limitation.
    4. Include relevant source URLs in your answer.
    5. Make the answer easy to understand.
    """

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"""
                User question:
                {question}

                Web search results:
                {context}

                Answer the user's question using these results.
                """
            )
        ]
    )

    print("\n--- WEB ANSWER GENERATED ---")

    return {
        "answer": response.content
    }


# =========================================================
# STEP 10: BUILD THE LANGGRAPH WORKFLOW
# =========================================================

agent = StateGraph(AgentState)


# Register nodes.
agent.add_node("classify_question", classify_question)
agent.add_node("approval_checker", approval_checker)
agent.add_node("chat_boat", chat_boat)
agent.add_node("web_search", web_search)


# Start by classifying the question.
agent.add_edge(
    START,
    "classify_question"
)


# Decide whether to ask for permission or answer directly.
agent.add_conditional_edges(
    "classify_question",
    route_after_classification,
    {
        "approval_checker": "approval_checker",
        "chat_boat": "chat_boat"
    }
)


# After approval, either search or answer directly.
agent.add_conditional_edges(
    "approval_checker",
    route_after_approval,
    {
        "web_search": "web_search",
        "chat_boat": "chat_boat"
    }
)


# Both answer paths finish the workflow.
agent.add_edge("chat_boat", END)
agent.add_edge("web_search", END)


# =========================================================
# STEP 11: COMPILE WITH CHECKPOINTING
# =========================================================

memory = MemorySaver()

app = agent.compile(
    checkpointer=memory
)


# =========================================================
# STEP 12: RUN THE AGENT
# =========================================================

def main():

    print("\n===== LANGGRAPH HITL AGENT =====")

    user_input = input(
        "\nAsk your question: "
    ).strip()

    if not user_input:
        print("Please enter a question.")
        return

    config = {
        "configurable": {
            "thread_id": "human-approval-001"
        }
    }

    # Start the graph.
    result = app.invoke(
        {
            "question": user_input
        },
        config=config
    )

    # Check whether LangGraph paused for human approval.
    interrupts = result.get("__interrupt__", [])

    if interrupts:

        interrupt_data = interrupts[0].value

        print("\n===== HUMAN APPROVAL REQUIRED =====")
        print(interrupt_data["message"])
        print("Question:", interrupt_data["question"])
        print("Reason:", interrupt_data["reason"])

        choice = input(
            "\nAllow web search? (yes/no): "
        ).strip().lower()

        # Resume the SAME graph execution.
        result = app.invoke(
            Command(
                resume=choice in ("yes", "y")
            ),
            config=config
        )

    # Display the final answer.
    print("\n===== FINAL ANSWER =====")

    print(
        result.get(
            "answer",
            "The agent did not produce an answer."
        )
    )


if __name__ == "__main__":
    main()