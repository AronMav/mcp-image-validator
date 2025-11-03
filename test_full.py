#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full functional test for MCP Image Validator.
Tests the complete flow with a real image.
"""

import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from ollama_vision_client import OllamaVisionClient


def test_full_flow():
    """Test complete image description flow"""
    print("=" * 60)
    print("MCP Image Validator - Full Functional Test")
    print("=" * 60)
    print()

    # Check API key
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        print("❌ OLLAMA_API_KEY not set!")
        print()
        print("Пожалуйста, настройте .env файл:")
        print("  1. cp .env.example .env")
        print("  2. Добавьте ваш OLLAMA_API_KEY")
        print("  3. Получить ключ: https://ollama.com/settings/keys")
        print()
        return False

    print("✓ API ключ найден")
    print()

    # Initialize client
    print("Инициализация клиента Ollama Cloud...")
    base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1")
    model = os.getenv("VISION_MODEL", "qwen3-vl:235b-cloud")

    try:
        client = OllamaVisionClient(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        print(f"✓ Клиент создан: {model}")
        print()
    except Exception as e:
        print(f"❌ Ошибка создания клиента: {e}")
        return False

    # Check connection
    print("Проверка подключения к Ollama Cloud...")
    success, message = client.check_connection()
    if success:
        print(f"✓ {message}")
        print()
    else:
        print(f"❌ {message}")
        print()
        return False

    # Find test image
    test_images = [
        "test.jpg",
        "test.png",
        "sample.jpg",
        "sample.png"
    ]

    test_image_path = None
    for img_name in test_images:
        img_path = Path(img_name)
        if img_path.exists():
            test_image_path = img_path
            break

    if not test_image_path:
        print("⚠️  Тестовое изображение не найдено")
        print()
        print("Для полного теста:")
        print("  1. Положите изображение (jpg/png) в директорию проекта")
        print("  2. Или укажите путь к изображению:")
        print()

        # Ask for image path
        try:
            user_path = input("Введите путь к изображению (или Enter для пропуска): ").strip()
            if user_path:
                user_path = user_path.strip('"\'')  # Remove quotes
                test_image_path = Path(user_path)
                if not test_image_path.exists():
                    print(f"❌ Файл не найден: {user_path}")
                    return False
            else:
                print("Тест пропущен (нет изображения)")
                return None
        except EOFError:
            print("Тест пропущен (нет изображения)")
            return None

    print(f"Найдено тестовое изображение: {test_image_path.name}")
    print()

    # Test image description
    print("Анализ изображения с помощью Qwen3-VL...")
    print("(Это может занять 10-30 секунд для модели на 235B параметров)")
    print()

    try:
        description = client.describe_image(
            image_path=str(test_image_path.absolute()),
            prompt="Describe this image in detail."
        )

        print("✓ Описание получено!")
        print()
        print("-" * 60)
        print("ОПИСАНИЕ ИЗОБРАЖЕНИЯ:")
        print("-" * 60)
        print(description)
        print("-" * 60)
        print()

        return True

    except Exception as e:
        print(f"❌ Ошибка при анализе изображения: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


def main():
    result = test_full_flow()

    print()
    print("=" * 60)

    if result is True:
        print("✓✓✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ! ✓✓✓")
        print()
        print("MCP Image Validator полностью функционален!")
        print()
        print("Готов к использованию в Claude Code:")
        print("  1. Добавьте сервер в настройки MCP")
        print("  2. Перезапустите Claude Code")
        print("  3. Используйте: 'Опиши изображение по пути ...'")
        print()
        return 0
    elif result is None:
        print("⚠️  Частичная проверка завершена")
        print()
        print("Базовые функции работают, но полный тест не выполнен.")
        print("Для полного теста укажите путь к изображению.")
        print()
        return 0
    else:
        print("❌ ТЕСТ НЕ ПРОЙДЕН")
        print()
        print("Проверьте:")
        print("  - API ключ Ollama Cloud действителен")
        print("  - Интернет соединение активно")
        print("  - Изображение доступно и корректно")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
