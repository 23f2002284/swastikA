from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from analyze_agent.prompts import MAIN_AGENT_PROMPT
from utils import make_system_prompt
from analyze_agent.tools import ALL_TOOLS, google_api_key
import logging
import asyncio
logger = logging.getLogger(__name__)

# IMPORTANT: Instantiate the model properly
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key
)

# If you really want create_react_agent:
from langgraph.prebuilt import create_react_agent

analyze_agent = create_react_agent(
    model=model,
    tools=ALL_TOOLS,
    prompt=make_system_prompt(MAIN_AGENT_PROMPT),
)

async def run_agent_async(user_input: str):
    """
    Async entrypoint for the agent.
    """
    try:
        # Create the event loop if it doesn't exist
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No running event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the agent
        result = await analyze_agent.ainvoke({"messages": user_input})
        return result
    except Exception as e:
        logger.error(f"Error in run_agent_async: {str(e)}", exc_info=True)
        raise

def run_agent(user_input: str):
    """
    Synchronous wrapper for the async agent.
    """
    try:
        return asyncio.run(run_agent_async(user_input))
    except Exception as e:
        logger.error(f"Error in run_agent: {str(e)}", exc_info=True)
        raise