from hello_agents import HelloAgentsLLM
from dotenv import load_dotenv

load_dotenv()

llm = HelloAgentsLLM(
    provider = "ollama",
    model = "qwen2.5:latest",
    base_url = "http://localhost:11434/v1",
    api_key = "ollama"
)

messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

# 发起调用，think等方法都已从父类继承，无需重写
response_stream = llm.think(messages)

print("Ollama Response:")
for chunk in response_stream:
    print(chunk, end="", flush=True)