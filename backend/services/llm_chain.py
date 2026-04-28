from __future__ import annotations
import os
from typing import List, Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from loguru import logger

SYSTEM_TEMPLATE = """\
You are a precise document analyst. Answer questions ONLY from the context below.

Rules:
- Use only what is in the provided context.
- If the answer is not in the context, say so explicitly.
- Cite page numbers or excerpts when they help the user.
- Use bullet points or numbered lists for multi-part answers.

Document context:
{context}
"""

QUESTION_TEMPLATE = """\
Based strictly on the context above, answer this question:

{question}
"""


def _build_context(chunks: List[dict]) -> str:
    if not chunks:
        return "No relevant context was found."
    parts = []
    for i, c in enumerate(chunks, 1):
        loc   = f"[{c.get('source', 'doc')} · p.{c['page']}]" if c.get("page") else f"[{c.get('source', 'doc')}]"
        parts.append(f"[Excerpt {i}] {loc}\n{c['content']}")
    return "\n\n".join(parts)


def _get_llm(temperature: float, max_tokens: int) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=api_key,
    )


def generate_answer(
    question:     str,
    chunks:       List[dict],
    history:      Optional[List[dict]] = None,
    temperature:  float = 0.2,
    max_tokens:   int   = 1024,
) -> tuple[str, str]:
    context  = _build_context(chunks)
    messages = [SystemMessage(content=SYSTEM_TEMPLATE.format(context=context))]

    for msg in (history or [])[-6:]:
        cls = HumanMessage if msg["role"] == "user" else AIMessage
        messages.append(cls(content=msg["content"]))

    messages.append(HumanMessage(content=QUESTION_TEMPLATE.format(question=question)))

    llm      = _get_llm(temperature, max_tokens)
    response = llm.invoke(messages)
    answer   = response.content.strip()

    logger.info(f"Answer generated | len={len(answer)}")
    return answer, "llama-3.1-8b-instant"