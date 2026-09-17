import time
import httpx
import asyncio
import logging
from typing import Optional

import ollama
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser

from max_assistant.utils.log_utils import log_banner

logger = logging.getLogger(__name__)


def validate_model_capabilities(
        model_name: str,
        base_url: str,
        required_capabilities: tuple[str, ...] = ("tools",),
) -> None:
    """
    Fails fast at startup if the configured model is missing or doesn't support
    what the reasoning graph needs (e.g. tool calling), instead of letting the
    container come up healthy and crash on the first user turn.
    """
    try:
        info = ollama.Client(host=base_url).show(model_name)
    except ollama.ResponseError as e:
        raise RuntimeError(
            f"OLLAMA_MODEL_NAME='{model_name}' could not be loaded from '{base_url}': {e}. "
            f"Check the model name/tag is correct and has been pulled (`ollama pull {model_name}`)."
        ) from e

    capabilities = set(info.capabilities or [])
    missing = [c for c in required_capabilities if c not in capabilities]
    if missing:
        raise RuntimeError(
            f"OLLAMA_MODEL_NAME='{model_name}' does not support required capability(ies) "
            f"{missing} (reports: {sorted(capabilities)}). The reasoning graph binds tools on "
            f"every interactive turn, so this model would crash on the first message. "
            f"Choose a model whose `ollama show {model_name}` output lists 'tools' under Capabilities."
        )

    logger.info(f"✅ Model '{model_name}' validated — capabilities: {sorted(capabilities)}")


def create_llm_instance(
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float | int = 120.0
) -> ChatOllama:
    """
    Synchronously initializes and returns a ChatOllama instance.
    """
    logger.info("=" * 50)
    logger.info("🚀 Initializing Ollama instance...")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Target: {base_url}")
    logger.info("=" * 50)

    validate_model_capabilities(model_name, base_url)

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        client_kwargs={"timeout": timeout},
    )
    return llm


async def preload_model_async(
        llm: ChatOllama,
        ready_event: Optional[asyncio.Event] = None,
        keep_alive: str = "-1",
        max_retries: int = 10,
        retry_delay: int = 2
):
    """
    Asynchronously preloads a model in Ollama with retry logic.
    This is designed to be run as a background task. It will set the
    provided asyncio.Event upon completion or failure.
    """
    parser = StrOutputParser()
    chain = llm | parser

    retries = 0
    start_time = time.monotonic()

    logger.info(f"🔥 Sending async warm-up request to load '{llm.model}' into memory.")
    try:
        while retries < max_retries:
            try:
                await chain.ainvoke(
                    "Hi",
                    config=RunnableConfig(configurable={"keep_alive": keep_alive})
                )

                end_time = time.monotonic()
                duration = end_time - start_time

                logger.info( log_banner(
                    [f"✅ Async warm-up complete! Model '{llm.model}' is ready.",
                     f"  Warm-up duration: {duration:.2f} seconds.           ",
                    ]) )
                return  # Warm-up successful

            except httpx.ConnectError:
                logging.warning("Connection to Ollama service failed during warm-up.")
                retries += 1
                if retries < max_retries:
                    logging.warning(f"Retrying warm-up in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(
                        f"Exceeded maximum retries for warm-up of '{llm.model}'. The model may not be preloaded.")
                    return
            except httpx.TimeoutException as e:
                retries += 1
                logger.warning(
                    f"⏳ Ollama warm-up request timed out ({type(e).__name__}) (Attempt {retries}/{max_retries}). "
                    f"Model '{llm.model}' is likely still cold-loading weights in the background. Retrying..."
                )
                if retries < max_retries:
                    await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"\n❌ FAILED TO WARM UP OLLAMA for model '{llm.model}'{type(e).__name__}", exc_info=True)
                logging.error(f"   Error: {e}")
                return
        logger.error(
            f"❌ Exceeded maximum retries ({max_retries}) for warm-up of '{llm.model}'. The model may not be preloaded."
        )
    finally:
        if ready_event:
            ready_event.set()
