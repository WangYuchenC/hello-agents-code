from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Memory:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type:str, content:str):
        """向记忆中添加一条新记录
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        logger.info(f"记忆已更新，新增一条 '{record_type}' 记录")

    def get_trajectory(self) -> str:
        """将所有记忆格式化为一个连贯的字符串文本，用于构建提示词.
        """
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(f"--- 上一轮尝试(代码) ---\n{record['content']}")
            elif record['type'] == 'reflection':
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}")
        
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最近一次的执行结果 (例如，最新生成的代码)。
        如果不存在，则返回 None。
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None