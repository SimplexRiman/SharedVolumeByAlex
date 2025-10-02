
# 🚀 Advanced Code Assistant with Qwen3 200B+ & LangGraph
# Мультиагентная система для помощи в разработке с использованием мощной локальной модели

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from pathlib import Path
import ast
from datetime import datetime

# Закомментированы импорты которые нужно будет установить
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama
# from langchain.tools import Tool
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.vectorstores import Chroma
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langgraph.graph import StateGraph, END, START
# from langgraph.prebuilt import ToolExecutor
# from langgraph.checkpoint.memory import MemorySaver

import subprocess
import tempfile
import re

class CodeAssistantState(TypedDict):
    """Состояние системы кода-помощника"""
    query: str
    code_context: str
    analysis_result: str
    generated_code: str
    test_code: str
    documentation: str
    refactored_code: str
    conversation_history: List[Dict]
    current_agent: str
    task_type: str
    language: str
    project_files: List[str]
    error_feedback: str

class Qwen3CodeAssistant:
    """
    Мультиагентная система помощи в коде на базе Qwen3 200B+

    Архитектура:
    - Supervisor Agent: Координирует работу специализированных агентов
    - Code Analyzer: Анализирует существующий код, находит проблемы
    - Code Generator: Генерирует новый код по требованиям
    - Test Generator: Создает unit тесты и integration тесты  
    - Documentation Agent: Генерирует документацию и комментарии
    - Refactoring Agent: Улучшает существующий код
    - RAG System: Поиск по кодовой базе для контекста
    """

    def __init__(self, 
                 model_name: str = "qwen3",  # Ваша модель без размера
                 ollama_base_url: str = "http://localhost:11434"):

        self.model_name = model_name
        self.ollama_base_url = ollama_base_url

        print(f"🚀 Инициализация Qwen3 Code Assistant...")
        print(f"📡 Модель: {model_name}")
        print(f"🔗 Ollama URL: {ollama_base_url}")

        # Настройки для Qwen3 (оптимальные параметры)
        self.qwen3_config = {
            "temperature": 0.7,       # Рекомендуется для Qwen3
            "top_p": 0.8,             # Оптимальный для кодирования
            "top_k": 20,              # Фокусировка на лучших токенах
            "repeat_penalty": 1.05,   # Избегаем повторений
            "num_ctx": 32768,         # Используем большой контекст
            "num_gpu": 1,             # GPU ускорение
            "num_thread": 8,          # CPU потоки
        }

        # Инициализация компонентов (здесь показана структура)
        self._setup_agents()
        self._setup_workflow()

        print("✅ Qwen3 Code Assistant готов к работе!")

    def _setup_agents(self):
        """Настройка промптов для специализированных агентов"""

        # Supervisor Agent - Главный координатор  
        self.supervisor_prompt = """
Ты - ведущий архитектор программного обеспечения, координирующий команду AI агентов.

Доступные эксперты:
🔍 **Code Analyzer** - анализ кода, поиск багов, code review
💻 **Code Generator** - создание нового кода, функций, классов
🧪 **Test Generator** - создание тестов (unit, integration)
📚 **Documentation Agent** - документация, комментарии
🔧 **Refactoring Agent** - улучшение и оптимизация кода

Запрос: {query}
Код: {code_context}

Определи:
1. TASK_TYPE: [analysis/generation/testing/documentation/refactoring]
2. AGENT: [analyzer/generator/tester/documenter/refactorer] 
3. REASONING: [почему выбран этот агент]
"""

        # Code Analyzer Agent
        self.analyzer_prompt = """
Ты - эксперт по анализу кода с глубоким пониманием лучших практик.

Код для анализа:
```{language}
{code_context}
```

Задача: {query}

Проведи детальный анализ:
1. ✅ **Корректность** - синтаксис, логические ошибки  
2. ⚡ **Производительность** - узкие места, оптимизации
3. 🔒 **Безопасность** - уязвимости, небезопасные паттерны
4. 📖 **Читаемость** - стиль, именование, структура
5. 🏗️ **Архитектура** - SOLID принципы, паттерны
6. 🧪 **Тестируемость** - возможности тестирования

Для каждой проблемы предложи конкретное решение с примером кода.
"""

        # Code Generator Agent
        self.generator_prompt = """
Ты - эксперт по генерации высококачественного, production-ready кода.

Требование: {query}
Язык: {language}
Контекст: {code_context}

Принципы генерации:
✅ Чистый, читаемый код
✅ Соответствие стандартам ({language} конвенции)
✅ Подробные docstrings/комментарии
✅ Обработка ошибок
✅ Типизация (type hints)
✅ Модульность

Структура ответа:
1. 📋 Описание решения
2. 💻 Полный код с комментариями  
3. 📖 Примеры использования
4. 🧪 Рекомендации по тестированию
"""

        # Test Generator Agent
        self.tester_prompt = """
Ты - эксперт по comprehensive тестированию ПО.

Код для тестирования:
```{language}
{code_context}
```

Задача: {query}

Создай полный набор тестов:
1. 🧪 **Unit тесты** - отдельные функции/методы
2. 🔗 **Integration тесты** - взаимодействие компонентов
3. 🎯 **Edge cases** - граничные случаи и ошибки
4. ⚡ **Performance тесты** - при необходимости

Для каждого теста:
- Описание что тестируется
- Setup/teardown если нужно
- Clear assertions с понятными сообщениями
- Позитивные и негативные сценарии

Используй подходящий фреймворк (pytest/Jest/JUnit).
"""

        # Documentation Agent
        self.documenter_prompt = """
Ты - технический писатель, специалист по документации кода.

Код для документирования:
```{language}
{code_context}
```

Задача: {query}

Создай подробную документацию:
1. 📚 **API Documentation** - функции, классы, методы
2. 📖 **Usage Examples** - практические примеры
3. 🏗️ **Architecture Overview** - общая архитектура
4. 🛠️ **Setup Instructions** - установка/настройка
5. 🔧 **Troubleshooting** - частые проблемы и решения

Для каждой функции/класса:
- Назначение и описание
- Параметры с типами
- Возвращаемые значения  
- Возможные исключения
- Примеры использования

Используй Markdown формат.
"""

        # Refactoring Agent
        self.refactorer_prompt = """
Ты - эксперт по рефакторингу с глубоким знанием паттернов проектирования.

Исходный код:
```{language}
{code_context}
```

Цель: {query}

Направления улучшения:
1. 🦨 **Code Smells** - устранение "дурно пахнущего" кода
2. 🔄 **DRY Principle** - устранение дублирования
3. 🏗️ **SOLID Principles** - улучшение архитектуры
4. ⚡ **Performance** - оптимизация производительности
5. 📖 **Readability** - улучшение читаемости
6. 🔧 **Maintainability** - упрощение поддержки

Для каждого изменения объясни:
- Что улучшается и почему
- Как влияет на код
- Альтернативные подходы

Предоставь отрефакторенный код с комментариями.
"""

        print("✅ Промпты агентов настроены")

    def _setup_workflow(self):
        """Настройка workflow (без LangGraph для демонстрации)"""

        self.agents = {
            "supervisor": self._supervisor_logic,
            "analyzer": self._analyzer_logic,
            "generator": self._generator_logic,
            "tester": self._tester_logic,
            "documenter": self._documenter_logic,
            "refactorer": self._refactorer_logic,
        }

        print("✅ Workflow настроен")

    def _make_ollama_request(self, prompt: str, system_prompt: str = "", temperature: float = None) -> str:
        """Запрос к Ollama через API"""

        import requests

        temp = temperature if temperature is not None else self.qwen3_config["temperature"]

        # Формируем запрос в формате Ollama API
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temp,
                "top_p": self.qwen3_config["top_p"],
                "top_k": self.qwen3_config["top_k"],
                "repeat_penalty": self.qwen3_config["repeat_penalty"],
                "num_ctx": self.qwen3_config["num_ctx"]
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 минут на генерацию
            )

            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            else:
                return f"Ошибка запроса: {response.status_code}"

        except Exception as e:
            return f"Ошибка соединения с Ollama: {e}"

    def _supervisor_logic(self, state: CodeAssistantState) -> CodeAssistantState:
        """Логика супервизора"""

        prompt = self.supervisor_prompt.format(
            query=state["query"],
            code_context=state["code_context"]
        )

        response = self._make_ollama_request(
            prompt, 
            "Ты координатор AI агентов для помощи в программировании.",
            temperature=0.3  # Меньше креативности для классификации
        )

        # Простой парсинг ответа
        task_type = self._extract_field(response, "TASK_TYPE") or "analysis"
        agent = self._extract_field(response, "AGENT") or "analyzer"
        reasoning = self._extract_field(response, "REASONING")

        state["task_type"] = task_type
        state["current_agent"] = agent
        state["conversation_history"].append({
            "role": "supervisor",
            "content": f"Задача: {task_type}, Агент: {agent}",
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })

        print(f"🎯 Супервизор -> Задача: {task_type} | Агент: {agent}")
        return state

    def _analyzer_logic(self, state: CodeAssistantState) -> CodeAssistantState:
        """Логика анализатора"""

        prompt = self.analyzer_prompt.format(
            query=state["query"],
            code_context=state["code_context"],
            language=state["language"]
        )

        response = self._make_ollama_request(
            prompt,
            "Ты эксперт по анализу кода и архитектуре ПО.",
            temperature=0.4
        )

        state["analysis_result"] = response
        state["conversation_history"].append({
            "role": "analyzer",
            "content": "Анализ кода завершен",
            "timestamp": datetime.now().isoformat()
        })

        print("🔍 Анализ кода завершен")
        return state

    def _generator_logic(self, state: CodeAssistantState) -> CodeAssistantState:
        """Логика генератора"""

        prompt = self.generator_prompt.format(
            query=state["query"],
            language=state["language"],
            code_context=state["code_context"]
        )

        response = self._make_ollama_request(
            prompt,
            "Ты эксперт по генерации высококачественного кода.",
            temperature=0.7  # Больше креативности для генерации
        )

        state["generated_code"] = response
        state["conversation_history"].append({
            "role": "generator",
            "content": "Код сгенерирован",
            "timestamp": datetime.now().isoformat()
        })

        print("💻 Код сгенерирован")
        return state

    def _tester_logic(self, state: CodeAssistantState) -> CodeAssistantState:
        """Логика создания тестов"""

        prompt = self.tester_prompt.format(
            query=state["query"],
            code_context=state["code_context"],
            language=state["language"]
        )

        response = self._make_ollama_request(
            prompt,
            "Ты эксперт по тестированию ПО.",
            temperature=0.5
        )

        state["test_code"] = response
        state["conversation_history"].append({
            "role": "tester",
            "content": "Тесты созданы",
            "timestamp": datetime.now().isoformat()
        })

        print("🧪 Тесты созданы")
        return state

    def _documenter_logic(self, state: CodeAssistantState) -> CodeAssistantState:
        """Логика создания документации"""

        prompt = self.documenter_prompt.format(
            query=state["query"],
            code_context=state["code_context"],
            language=state["language"]
        )

        response = self._make_ollama_request(
            prompt,
            "Ты технический писатель, эксперт по документации кода.",
            temperature=0.5
        )

        state["documentation"] = response
        state["conversation_history"].append({
            "role": "documenter",
            "content": "Документация создана",
            "timestamp": datetime.now().isoformat()
        })

        print("📚 Документация создана")
        return state

    def _refactorer_logic(self, state: CodeAssistantState) -> CodeAssistantState:
        """Логика рефакторинга"""

        prompt = self.refactorer_prompt.format(
            query=state["query"],
            code_context=state["code_context"],
            language=state["language"]
        )

        response = self._make_ollama_request(
            prompt,
            "Ты эксперт по рефакторингу кода и паттернам проектирования.",
            temperature=0.6
        )

        state["refactored_code"] = response
        state["conversation_history"].append({
            "role": "refactorer", 
            "content": "Код отрефакторен",
            "timestamp": datetime.now().isoformat()
        })

        print("🔧 Рефакторинг завершен")
        return state

    def _extract_field(self, text: str, field_name: str) -> str:
        """Извлечение поля из ответа"""
        patterns = [
            f"{field_name}:\s*(.*?)(?=\n[A-Z_]+:|$)",
            f"{field_name}\s*=\s*(.*?)(?=\n|$)",
            f"\*\*{field_name}\*\*:\s*(.*?)(?=\n|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        return ""

    def ask(self, query: str, code_context: str = "", language: str = "python") -> Dict[str, Any]:
        """Главный метод для взаимодействия с ассистентом"""

        print(f"\n🤖 Обработка запроса: {query[:100]}...")

        # Начальное состояние
        state = CodeAssistantState(
            query=query,
            code_context=code_context,
            analysis_result="",
            generated_code="",
            test_code="",
            documentation="",
            refactored_code="",
            conversation_history=[],
            current_agent="supervisor",
            task_type="",
            language=language,
            project_files=[],
            error_feedback=""
        )

        try:
            # 1. Супервизор определяет задачу
            state = self._supervisor_logic(state)

            # 2. Выполняем задачу соответствующим агентом
            agent_name = state["current_agent"]
            if agent_name in self.agents:
                state = self.agents[agent_name](state)

            # Формируем результат
            result = {
                "query": query,
                "task_type": state["task_type"],
                "agent_used": agent_name,
                "analysis": state["analysis_result"],
                "generated_code": state["generated_code"],
                "tests": state["test_code"],
                "documentation": state["documentation"],
                "refactored_code": state["refactored_code"],
                "conversation_history": state["conversation_history"],
                "timestamp": datetime.now().isoformat()
            }

            print("✅ Запрос обработан успешно")
            return result

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return {
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

    def run_interactive(self):
        """Интерактивная сессия"""

        print(f"""
🚀 Qwen3 Code Assistant готов к работе!
   Модель: {self.model_name}

💡 Примеры запросов:
   • "Создай функцию для сортировки массива"
   • "Проанализируй этот код на предмет безопасности"
   • "Напиши тесты для класса User"
   • "Создай документацию для API"
   • "Отрефактори этот код"

📝 Команды:
   /help - справка
   /exit - выход
   /test - тестовый запрос
        """)

        while True:
            try:
                user_input = input("\n👤 Ваш запрос: ").strip()

                if user_input.lower() == '/exit':
                    print("👋 До свидания!")
                    break

                elif user_input.lower() == '/help':
                    print(self._get_help_text())
                    continue

                elif user_input.lower() == '/test':
                    self._run_test_examples()
                    continue

                # Обычный запрос
                result = self.ask(user_input)
                self._display_result(result)

            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")

    def _get_help_text(self) -> str:
        return """
📋 Справка по использованию Qwen3 Code Assistant

🎯 Типы задач:
   • Анализ кода - поиск багов, проблем производительности, безопасности
   • Генерация кода - создание функций, классов, модулей по описанию
   • Создание тестов - unit тесты, integration тесты
   • Документация - docstrings, README, API документация
   • Рефакторинг - улучшение существующего кода

💬 Как задавать вопросы:
   • Будьте конкретны: "Создай класс User с методами авторизации"
   • Указывайте язык: "Напиши на Java функцию сортировки"
   • Приводите контекст: "В Django проекте нужен API для..."

📖 Примеры:
   • "Анализируй код: [вставить код]"
   • "Создай REST API для управления пользователями"
   • "Напиши тесты для функции fibonacci"
   • "Документируй этот класс"
        """

    def _run_test_examples(self):
        """Запуск тестовых примеров"""

        examples = [
            {
                "query": "Создай функцию для вычисления числа Фибоначчи",
                "desc": "Генерация кода"
            },
            {
                "query": "Проанализируй этот код",
                "code": "def login(user, pwd): return user == 'admin' and pwd == '123'",
                "desc": "Анализ безопасности"
            },
            {
                "query": "Создай тесты для функции сложения",
                "code": "def add(a, b): return a + b",
                "desc": "Генерация тестов"
            }
        ]

        print("\n🧪 Запуск тестовых примеров...")

        for i, example in enumerate(examples, 1):
            print(f"\n{'='*50}")
            print(f"Пример {i}: {example['desc']}")
            print(f"{'='*50}")

            code = example.get('code', '')
            result = self.ask(example['query'], code)

            if 'error' not in result:
                print(f"✅ Успешно: {result['task_type']} через {result['agent_used']}")
            else:
                print(f"❌ Ошибка: {result['error']}")

    def _display_result(self, result: Dict[str, Any]):
        """Отображение результатов"""

        if "error" in result:
            print(f"❌ Ошибка: {result['error']}")
            return

        print(f"\n🎯 Задача: {result['task_type']}")
        print(f"🤖 Агент: {result['agent_used']}")

        # Показываем результаты работы разных агентов
        if result.get('analysis'):
            print(f"\n🔍 АНАЛИЗ КОДА:")
            print("-" * 40)
            print(result['analysis'][:1000] + ("..." if len(result['analysis']) > 1000 else ""))

        if result.get('generated_code'):
            print(f"\n💻 СГЕНЕРИРОВАННЫЙ КОД:")
            print("-" * 40)
            print(result['generated_code'])

        if result.get('tests'):
            print(f"\n🧪 ТЕСТЫ:")
            print("-" * 40)
            print(result['tests'])

        if result.get('documentation'):
            print(f"\n📚 ДОКУМЕНТАЦИЯ:")
            print("-" * 40)
            print(result['documentation'])

        if result.get('refactored_code'):
            print(f"\n🔧 ОТРЕФАКТОРЕННЫЙ КОД:")
            print("-" * 40)
            print(result['refactored_code'])

# Основная функция запуска
def main():
    """Главная функция для запуска ассистента"""

    print("""
🚀 QWEN3 CODE ASSISTANT
======================
Мультиагентная система помощи в разработке

Убедитесь что:
✅ Ollama запущен (ollama serve)
✅ Модель Qwen3 загружена (ollama pull qwen3)
✅ Модель доступна на localhost:11434
    """)

    # Можете изменить имя модели на ваше
    model_name = input("🤖 Введите имя модели (по умолчанию 'qwen3'): ").strip() or "qwen3"

    try:
        assistant = Qwen3CodeAssistant(model_name=model_name)
        assistant.run_interactive()
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        print("\nПроверьте:")
        print("• Запущен ли Ollama (ollama serve)")
        print("• Доступна ли модель (ollama list)")
        print("• Корректно ли имя модели")

if __name__ == "__main__":
    main()
