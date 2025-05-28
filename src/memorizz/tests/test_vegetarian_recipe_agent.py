import os
import sys
import pytest
from dotenv import load_dotenv
from scenario import Scenario, TestingAgent, scenario_cache

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
load_dotenv()

from ..memagent import MemAgent
from ..memory_provider.mongodb.provider import MongoDBConfig, MongoDBProvider



# Create a memory provider
mongodb_config = MongoDBConfig(uri=os.environ["MONGODB_URI"])
memory_provider = MongoDBProvider(mongodb_config)

Scenario.configure(testing_agent=TestingAgent(model="openai/gpt-4o-mini"))

mem_agent = MemAgent(memory_provider=memory_provider)

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_vegetarian_recipe_agent():
    agent = mem_agent

    def vegetarian_recipe_agent(message, context):
        # Call your agent here
        response = agent.run(message)
        return {
            "message": response
        }

    # Define the scenario
    scenario = Scenario(
        "User is looking for a dinner idea",
        agent=vegetarian_recipe_agent,
        success_criteria=[
            "Recipe agent generates a vegetarian recipe",
            "Recipe includes a list of ingredients",
            "Recipe includes step-by-step cooking instructions",
        ],
        failure_criteria=[
            "The recipe is not vegetarian or includes meat",
            "The agent asks more than two follow-up questions",
        ],
    )

    # Run the scenario and get results
    result = await scenario.run()

    # Assert for pytest to know whether the test passed
    assert result.success
