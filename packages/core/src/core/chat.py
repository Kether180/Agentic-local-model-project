"""Chat models, via LangChain's unified provider interface.

Steps ask for a size (`small`/`medium`/`large`); config maps that to a provider-specific model id.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from core.config import ModelSize, settings


def get_chat_model(size: ModelSize = "small") -> BaseChatModel:
    cfg = settings.chat
    if cfg.provider == "ollama":
        return init_chat_model(
            model=cfg.chat_models[size],
            model_provider="ollama",
            base_url=cfg.base_url,
            reasoning=cfg.reasoning,
            num_predict=cfg.num_predict,
            num_ctx=cfg.num_ctx,
        )
    return init_chat_model(
        model=cfg.chat_models[size], model_provider="bedrock_converse", region_name=cfg.region
    )
