import base64
import asyncio
from typing import Optional
from openai import AsyncOpenAI
from src.config import settings

# Создаём клиент DeepSeek
client = AsyncOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


async def analyze_food_input(
    text_description: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    images_bytes: Optional[List[bytes]] = None,
    language: str = "ru",
) -> FoodAnalysisResponse:
    """
    Отправляет фото еды в DeepSeek и получает анализ КБЖУ.
    """
    # Кодируем изображение в base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{base64_image}"

    try:
        response = await client.chat.completions.create(
            model="deepseek-v4.1-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Ты — диетолог. Проанализируй фото блюда. "
                                "Определи, что это за еда, и оцени примерный вес порции. "
                                "Посчитай калории, белки, жиры и углеводы (КБЖУ). "
                                "Ответь строго в формате JSON: "
                                '{"dish": "название", "weight_g": 200, "calories": 350, '
                                '"protein": 15, "fat": 10, "carbs": 45}'
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f'{{"error": "Не удалось проанализировать фото: {str(e)}"}}'
