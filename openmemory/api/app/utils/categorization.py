import json
import logging

from openai import OpenAI
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT

load_dotenv()

openai_client = OpenAI()


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )

        content = response.choices[0].message.content
        response_json = json.loads(content)
        categories = response_json['categories']
        categories = [cat.strip().lower() for cat in categories]

        return categories

    except Exception as e:
        print(f"[ERROR] Failed to get categories: {e}")
        try:
            print(f"[DEBUG] Raw response: {content}")
        except Exception as debug_e:
            print(f"[DEBUG] Could not extract raw response: {debug_e}")
        raise