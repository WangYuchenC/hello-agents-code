import os
from typing import Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    default_model: str = "deepseek-v4-flash"
    default_provider: str = "deepseek"
    temperature: float = 0.2
    max_tokens: int | None = None

    debug: bool = False
    log_level: str = "INFO"

    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> 'Config':
        if max_tokens := os.getenv("MAX_TOKENS"):
            if max_tokens != '':
                max_tokens = int(max_tokens)
            else:
                max_tokens = None

        return cls(
            debug = os.getenv("DEBUG", 'false').lower() == "true",
            log_level = os.getenv('LOG_LEVEL', 'INFO'),
            temperature = float(os.getenv("TEMPERATURE", 0.2)),
            max_tokens = max_tokens # type: ignore
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()