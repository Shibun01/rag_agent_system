from functools import lru_cache
from openai import AsyncAzureOpenAI
from app.config.settings import get_settings

settings = get_settings()


@lru_cache
def get_azure_openai_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


async def chat_completion(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    tools: list[dict] | None = None,
) -> str:
    client = get_azure_openai_client()
    kwargs = dict(
        model=settings.azure_openai_deployment_name,
        messages=messages,
        max_completion_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message


async def get_embedding(text: str) -> list[float]:
    client = get_azure_openai_client()
    print(f"Checking with the embedding LLM: {client}")
    response = await client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=text,
    )
    return response.data[0].embedding


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    client = get_azure_openai_client()
    response = await client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=texts,
    )
    return [item.embedding for item in response.data]
