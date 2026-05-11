import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class HelloAgentsLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, api_key, base_url]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        verbose: bool = False,
    ) -> str | None:
        logger.debug(f"正在调用{self.model}模型...")
        try:
            res = self.client.chat.completions.create(
                model=self.model,  # type: ignore
                messages=messages,  # type: ignore
                temperature=temperature,
                stream=True,
            )  # type: ignore

            logger.debug("大语言模型响应成功:")
            collected_content = []
            for chunk in res:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content or ""
                if verbose:
                    print(content, end="", flush=True)
                collected_content.append(content)
            if verbose:
                print()
            return "".join(collected_content)
        except Exception as e:
            logger.error(f"调用LLM API时发生错误: {e}")
            return None


if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes Python code.",
            },
            {"role": "user", "content": "写一个快速排序算法"},
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages, verbose=True)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
