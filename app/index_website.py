# ─────────────────────────────────────
# Imports and Config
# ─────────────────────────────────────
import os
import pickle
from datetime import datetime
from enum import Enum, auto
from urllib.parse import urlparse

import pandas as pd
from config.setttings import get_settings
from database.vector_store import VectorStore
from firecrawl import FirecrawlApp, ScrapeOptions
from llama_index.core.node_parser import MarkdownNodeParser, SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
from timescale_vector.client import uuid_from_time

# ─────────────────────────────────────
# Constants and Settings
# ─────────────────────────────────────
settings = get_settings()
api_key = settings.firecrawl.api_key
openai_key = settings.openai.api_key
app = FirecrawlApp(api_key=api_key)
vec = VectorStore()
# vec.delete(delete_all=True)


# ─────────────────────────────────────
# Step 1: Crawl
# ─────────────────────────────────────
def crawl_site(url: str, save_dir: str):
    result = app.crawl_url(
        url=url,
        limit=25,
        scrape_options=ScrapeOptions(formats=["markdown"]),
        poll_interval=30,
    )
    if not result.success:
        raise RuntimeError(result.error)

    os.makedirs(save_dir, exist_ok=True)

    for doc in result.data:
        with open(f"{save_dir}/{doc.metadata['scrapeId']}.pkl", "wb") as f:
            pickle.dump(doc, f)


def crawl_url(url: str, save_dir: str):
    """
    Crawl a website and return the response.
    Args:
        url (str): The URL of the website to crawl.
    Returns:
        dict: The response from the website.
    """
    # Install with pip install firecrawl-py

    result = app.crawl_url(
        url=url,
        limit=25,
        scrape_options=ScrapeOptions(formats=["markdown"]),
        poll_interval=30,
    )

    if not result.success:
        raise RuntimeError(result.error)

    os.makedirs(save_dir, exist_ok=True)

    for doc in result.data:
        with open(f"{save_dir}/{doc.metadata['scrapeId']}.pkl", "wb") as f:
            pickle.dump(doc, f)


# ─────────────────────────────────────
# Step 2: Chunk
# ─────────────────────────────────────
class SplitMethod(Enum):
    SENTENCE = auto()
    SEMANTIC = auto()


def truncate_after_phrase(text: str, phrase: str) -> str:
    before, sep, after = text.partition(phrase)
    return before  # Includes the phrase itself


def pre_clean(text: str) -> Document:
    text = text.replace("[0](https://www.neuralaspect.com/cart)\n\n", "")
    text = truncate_after_phrase(text, "#### Contact")
    text = truncate_after_phrase(
        text,
        "\n[Gareth Davies](https://www.neuralaspect.com/posts?author=678a4a2fbfbaf269ea70a4f2)\n\nGareth is an AI researcher",
    )
    return text


def chunk_documents(input_dir: str, method: SplitMethod):
    nodes = []
    for fname in os.listdir(input_dir):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(input_dir, fname), "rb") as f:
            raw = pickle.load(f)

        raw.markdown = pre_clean(raw.markdown)

        doc = Document(
            text=raw.markdown, metadata=raw.metadata, doc_id=raw.metadata["url"]
        )
        splitter = (
            MarkdownNodeParser()
            if method == SplitMethod.SENTENCE
            else SemanticSplitterNodeParser(embed_model=OpenAIEmbedding())
        )
        nodes.extend(splitter.get_nodes_from_documents([doc]))

    return nodes


# ─────────────────────────────────────
# Step 3: Annotate with LLM
# ─────────────────────────────────────
from pydantic import BaseModel, Field


class QuestionAnswerResponse(BaseModel):
    prompt: str = Field(
        description="The prompt representing the question asked by the user"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to provide a realistic prompt"
    )


import asyncio

from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=openai_key)


async def annotate_node(node):
    try:
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You're a helpful assistant. The context is a response to a 
                    question from an LLM. Your task is to provide the prompt that generated the context. 
                    The prompts were made in British English.""",
                },
                {"role": "user", "content": node.text},
            ],
            response_format=QuestionAnswerResponse,  # you can parse manually unless using a special schema
        )

        event = completion.choices[0].message.parsed
        node.text = f"Prompt: {event.prompt}\nAnswer: {node.text}"
        node.metadata["enough_context"] = event.enough_context
        return node

    except Exception as e:
        print(f"Error annotating node: {e}")
        return node  # return unmodified if there's an error


async def annotate_nodes_async(nodes):
    tasks = [annotate_node(node) for node in nodes]
    return await asyncio.gather(*tasks)


# Entry point if you're calling from sync code
def annotate_nodes(nodes):
    return asyncio.run(annotate_nodes_async(nodes))


# ─────────────────────────────────────
# Step 4: Embedding
# ─────────────────────────────────────


from more_itertools import chunked
from openai import OpenAI


def embed_nodes(nodes, batch_size=100):
    records = []
    texts = [node.text.replace("\n", " ") for node in nodes]
    node_batches = list(chunked(list(zip(nodes, texts)), batch_size))

    for batch in node_batches:
        nodes_batch, texts_batch = zip(*batch)

        embeddings = vec.openai_client.embeddings.create(
            input=list(texts_batch),
            model=vec.embedding_model,
        ).data

        for node, embedding_obj in zip(nodes_batch, embeddings):
            embedding = embedding_obj.embedding
            records.append(
                {
                    "id": str(uuid_from_time(datetime.now())),
                    "metadata": {**{"category": "post"}, **node.metadata},
                    "contents": node.text,
                    "embedding": embedding,
                }
            )
    return pd.DataFrame(records)


# ─────────────────────────────────────
# Step 5: Store
# ─────────────────────────────────────
def store_embeddings(df: pd.DataFrame):
    vec.create_tables()
    vec.create_index()  # DiskANN index
    vec.create_keyword_search_index()  # GIN index
    vec.upsert(df)


# ─────────────────────────────────────
# Main Pipeline Entry
# ─────────────────────────────────────
def main():
    url = "https://www.neuralaspect.com/posts/how-saas-will-die/"
    domain = urlparse(url).netloc
    current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD

    # save_dir = f"../data/date={current_date}/domain={domain}"
    save_dir = "/Users/garethdavies/Development/workspaces/postgres_rag/data/date=2025-04-24/domain=www.neuralaspect.com"

    # crawl_url(url, save_dir)
    nodes = chunk_documents(save_dir, SplitMethod.SENTENCE)
    # nodes = annotate_nodes(nodes)
    df = embed_nodes(nodes)
    store_embeddings(df)
    return nodes


if __name__ == "__main__":
    node = main()
