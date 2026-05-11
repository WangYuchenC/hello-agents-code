import os
from serpapi import SerpApiClient
from typing import Dict, Callable
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def search(query: str) -> str:
    """一个基于SerpApi的网页搜索引擎工具.
    它能智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    logger.debug(f"正在执行[SerpAPI]网页搜搜: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY未在.env文件中配置"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn",  # 语言代码
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


@dataclass
class Tool:
    description: str
    func: Callable


class ToolExcutor:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, name: str, description: str, func: Callable):
        """向工具箱中注册一个新工具."""
        if name in self.tools:
            logger.warning(f"工具 '{name}' 已存在，将被覆盖.")
        self.tools[name] = Tool(description=description, func=func)
        logger.info(f"注册工具 '{name}' ")

    def get_tool(self, name: str) -> Tool | None:
        """根据名称获取一个工具的执行函数."""
        return self.tools.get(name)

    def get_available_tools(self) -> str:
        """获取所有可用工具的格式化描述字符串"""
        return "\n".join(
            [f"- {name}:  {tool.description}" for name, tool in self.tools.items()]
        )
