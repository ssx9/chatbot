from httpx import AsyncClient
from pydantic_ai import Agent
from dotenv import load_dotenv
from pydantic_ai.capabilities import Thinking, WebSearch

from deps import Deps
from tools import get_lat_lng, get_weather

# 加载 .env 文件
load_dotenv()


instructions = 'Be concise, reply with one sentence.'
# instructions = 'You are a helpful assistant.'
weather_agent = Agent(
    'deepseek:deepseek-v4-pro',
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    instructions=instructions,
    deps_type=Deps,
    retries=2,
    capabilities=[],
)

weather_agent.tool(get_lat_lng)
weather_agent.tool(get_weather)

client = AsyncClient()
deps = Deps(client=client)
