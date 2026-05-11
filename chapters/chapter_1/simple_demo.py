import logging
import requests
import os
from tavily import TavilyClient
import re
import dotenv

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(filename)s %(levelname)s %(message)s"
)
dotenv.load_dotenv()

MODEL_ID = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询制定城市的实时天气。
- `get_attraction(city: str, weather:str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求：
你的每次恢复必须严格遵循以下格式，包含一对Thought和Action:

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一:
1. 调用工具: function_name(arg_name="arg_value")
2. 结束任务: Finish[最终答案]

# 重要提示:
- 每次只输出一对Though-Action
- Action必须在同一行，不能换行
- 当收集到足够信息可以回答用户问题时，必须使用Action: Finish[最终答案]格式结束

请开始吧!
"""


class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        logging.info("正在调用大语言模型....")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            res = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
            answer = res.choices[0].message.content
            logging.info("大语言模型响应成功")
            return answer
        except Exception as e:
            logging.error(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言服务时出错."


def get_weather(city: str) -> str:
    """
    通过调用wttr.in API查询真实的天气信息
    """
    url = f"https://wttr.in/{city}?format=j1"

    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        logging.debug(f"Raw response of weather query: {data}")
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        return f"{city}当前天气:{weather_desc}, 气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐"""

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量"

    tavily = TavilyClient(api_key=api_key)

    query = f"'{city}' 在'{weather}'天气下最值得去的旅游经典推荐及理由"

    try:
        res = tavily.search(query=query, search_depth="basic", include_answer=True)
        if res.get("answer"):
            return res["answer"]

        formatted_results = []
        for result in res.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游经典推荐"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


available_tools = {"get_weather": get_weather, "get_attraction": get_attraction}


llm = OpenAICompatibleClient(
    model=MODEL_ID, api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL
)

user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

logging.info(f"用户输入: {user_prompt}\n" + "=" * 40)

for i in range(5):
    logging.info(f"--- 循环 {i + 1} ---\n")
    full_prompt = "\n".join(prompt_history)

    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    logging.debug(f"LLM raw output: {llm_output}")
    match = re.search(
        r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
        llm_output,
        re.DOTALL,
    )
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    logging.info(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        logging.error(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        logging.info(f"任务完成，最终答案: {final_answer}")
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    logging.info(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)
