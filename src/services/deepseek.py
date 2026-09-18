import base64
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from src.config import settings


class FoodItem(BaseModel):
    name: str
    portion: str
    calories: float
    protein: float
    fat: float
    carb: float


class FoodAnalysisResponse(BaseModel):
    food_items: List[FoodItem] = Field(default_factory=list)
    total_calories: float = 0
    total_protein: float = 0
    total_fat: float = 0
    total_carb: float = 0


client = AsyncOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-v4.1-flash"

SYSTEM_PROMPT = """Ты — профессиональный диетолог. Анализируй еду на фото или в тексте.
Отвечай СТРОГО в формате JSON без markdown-обёрток:
{
  "food_items": [
    {"name": "название блюда", "portion": "вес/порция", "calories": 100, "protein": 5, "fat": 3, "carb": 10}
  ],
  "total_calories": 100,
  "total_protein": 5,
  "total_fat": 3,
  "total_carb": 10
}"""


async def _call_deepseek(messages: list) -> FoodAnalysisResponse:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    return FoodAnalysisResponse(**data)


async def analyze_food_input(
    text_description: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    images_bytes: Optional[List[bytes]] = None,
    language: str = "ru",
) -> FoodAnalysisResponse:
    content = [{"type": "text", "text": SYSTEM_PROMPT}]

    user_text = text_description or "Проанализируй это блюдо."
    content.append({"type": "text", "text": user_text})

    all_images = []
    if images_bytes:
        all_images.extend(images_bytes)
    if image_bytes:
        all_images.append(image_bytes)

    for img in all_images:
        b64 = base64.b64encode(img).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    messages = [{"role": "user", "content": content}]
    return await _call_deepseek(messages)


async def adjust_food_analysis(
    original_data: dict,
    correction_text: str,
    language: str = "ru",
) -> FoodAnalysisResponse:
    prompt = (
        f"Вот текущий анализ блюда:\n{json.dumps(original_data, ensure_ascii=False)}\n\n"
        f"Пользователь хочет исправить: {correction_text}\n\n"
        f"Верни обновлённый анализ в том же JSON-формате."
    )
    messages = [{"role": "user", "content": prompt}]
    return await _call_deepseek(messages)
