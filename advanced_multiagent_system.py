
# Продвинутая архитектура мультиагентной системы для анализа кода
# Версия с улучшенной модульностью и производительностью

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import chromadb
from sentence_transformers import SentenceTransformer
import tree_sitter_python as ts_python
from tree_sitter import Language, Parser

# НАСТРОЙКА ЛОГИРОВАНИЯ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# КОНФИГУРАЦИЯ СИСТЕМЫ
@dataclass
class SystemConfig:
    """Конфигурация системы"""
    project_path: str
    vector_db_path: str = "./code_embeddings"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "ollama:qwen2.5-coder:7b"
    max_chunk_size: int = 500
    chunk_overlap: int = 50
    max_concurrent_agents: int = 3
    context_window: int = 32768

# АБСТРАКТНЫЕ ИНТЕРФЕЙСЫ

class BaseCodeAnalyzer(ABC):
    """Базовый интерфейс для анализаторов кода"""

    @abstractmethod
    def parse_file(self, file_path: str) -> List[Dict]:
        pass

    @abstractmethod
    def extract_metadata(self, code: str) -> Dict:
        pass

class BaseVectorStore(ABC):
    """Базовый интерфейс для векторных хранилищ"""

    @abstractmethod
    def add_documents(self, documents: List[Dict]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict]:
        pass

# РЕАЛИЗАЦИИ

class TreeSitterAnalyzer(BaseCodeAnalyzer):
    """Анализатор кода с использованием Tree-sitter"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.parsers = {}
        self._init_parsers()

    def _init_parsers(self):
        """Инициализация парсеров для разных языков"""
        try:
            self.parsers['python'] = {
                'language': Language(ts_python.language()),
                'parser': Parser(Language(ts_python.language())),
                'extensions': ['.py']
            }
            logger.info("Tree-sitter парсеры инициализированы")
        except Exception as e:
            logger.error(f"Ошибка инициализации парсеров: {e}")

    def parse_file(self, file_path: str) -> List[Dict]:
        """Парсинг файла и извлечение структурных элементов"""
        file_ext = Path(file_path).suffix

        # Определение языка по расширению
        language_info = None
        for lang, info in self.parsers.items():
            if file_ext in info['extensions']:
                language_info = info
                break

        if not language_info:
            return []

        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()

            tree = language_info['parser'].parse(source_code)
            chunks = []

            self._extract_code_elements(
                tree.root_node, source_code, chunks, file_path
            )

            return chunks

        except Exception as e:
            logger.error(f"Ошибка парсинга {file_path}: {e}")
            return []

    def _extract_code_elements(self, node, source, chunks, file_path):
        """Рекурсивное извлечение элементов кода"""
        important_types = [
            'function_definition', 'class_definition', 
            'method_definition', 'import_statement'
        ]

        if node.type in important_types:
            code_text = source[node.start_byte:node.end_byte].decode('utf-8')

            metadata = self.extract_metadata(code_text)
            metadata.update({
                'type': node.type,
                'file_path': file_path,
                'start_line': node.start_point[0],
                'end_line': node.end_point[0],
                'size': len(code_text)
            })

            chunks.append({
                'code': code_text,
                'metadata': metadata
            })

        for child in node.children:
            self._extract_code_elements(child, source, chunks, file_path)

    def extract_metadata(self, code: str) -> Dict:
        """Извлечение метаданных из кода"""
        metadata = {
            'imports': [],
            'functions': [],
            'classes': [],
            'complexity_estimate': 'low'
        }

        lines = code.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                metadata['imports'].append(line)
            elif line.startswith('def '):
                func_name = line.split('(')[0].replace('def ', '')
                metadata['functions'].append(func_name)
            elif line.startswith('class '):
                class_name = line.split('(')[0].split(':')[0].replace('class ', '')
                metadata['classes'].append(class_name)

        # Простая оценка сложности
        if len(lines) > 50 or 'for ' in code or 'while ' in code:
            metadata['complexity_estimate'] = 'high'
        elif len(lines) > 20:
            metadata['complexity_estimate'] = 'medium'

        return metadata

class ChromaVectorStore(BaseVectorStore):
    """Векторное хранилище на основе ChromaDB"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)

        self.client = chromadb.PersistentClient(path=config.vector_db_path)
        self.collection = self.client.get_or_create_collection(
            name="codebase_knowledge",
            metadata={"description": "Code analysis and generation knowledge base"}
        )

        logger.info("ChromaDB векторное хранилище инициализировано")

    def add_documents(self, documents: List[Dict]) -> None:
        """Добавление документов в векторное хранилище"""
        embeddings = []
        doc_texts = []
        metadatas = []
        ids = []

        for i, doc in enumerate(documents):
            # Создание комбинированного текста для embedding
            code = doc['code']
            metadata = doc['metadata']

            # Генерация описания кода
            description = self._generate_code_description(code, metadata)
            combined_text = f"{description}\n\n{code}"

            # Создание embedding
            embedding = self.embedding_model.encode(combined_text)

            embeddings.append(embedding.tolist())
            doc_texts.append(code)
            metadatas.append(metadata)
            ids.append(f"doc_{i}_{metadata.get('file_path', 'unknown')}")

        # Добавление в ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=doc_texts,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Добавлено {len(documents)} документов в векторное хранилище")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Поиск релевантных документов"""
        query_embedding = self.embedding_model.encode(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }

    def _generate_code_description(self, code: str, metadata: Dict) -> str:
        """Генерация естественно-языкового описания кода"""
        descriptions = []

        if metadata.get('type') == 'function_definition':
            func_names = metadata.get('functions', [])
            if func_names:
                descriptions.append(f"Функция {func_names[0]}")
        elif metadata.get('type') == 'class_definition':
            class_names = metadata.get('classes', [])
            if class_names:
                descriptions.append(f"Класс {class_names[0]}")

        if metadata.get('imports'):
            descriptions.append(f"Использует библиотеки: {', '.join(metadata['imports'][:3])}")

        if metadata.get('complexity_estimate') == 'high':
            descriptions.append("Сложная логика")

        return '. '.join(descriptions) if descriptions else "Код программы"

# СИСТЕМА УПРАВЛЕНИЯ АГЕНТАМИ

class AgentManager:
    """Менеджер для управления всеми агентами системы"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.code_analyzer = TreeSitterAnalyzer(config)
        self.vector_store = ChromaVectorStore(config)
        self.agents = {}
        self.supervisor = None

        self._init_agents()
        self._init_supervisor()

    def _init_agents(self):
        """Инициализация всех специализированных агентов"""

        # Агент анализа кода
        def analyze_codebase(query: str) -> str:
            """Анализ кодовой базы"""
            results = self.vector_store.search(query, k=5)

            analysis = f"Анализ кодовой базы для запроса: '{query}'\n\n"

            if results['documents']:
                analysis += "Найденные релевантные участки кода:\n"
                for i, (doc, meta, dist) in enumerate(zip(
                    results['documents'][:3], 
                    results['metadatas'][:3],
                    results['distances'][:3]
                )):
                    analysis += f"\n{i+1}. Файл: {meta.get('file_path', 'unknown')}\n"
                    analysis += f"   Тип: {meta.get('type', 'unknown')}\n"
                    analysis += f"   Релевантность: {1-dist:.2f}\n"
                    analysis += f"   Код: {doc[:150]}...\n"
            else:
                analysis += "Релевантный код не найден."

            return analysis

        self.agents['code_analyzer'] = create_react_agent(
            model=self.config.llm_model,
            tools=[analyze_codebase],
            prompt=(
                "Вы - эксперт по анализу кода и архитектуры проектов. "
                "Используйте инструмент analyze_codebase для поиска и анализа "
                "релевантного кода в проекте. Предоставляйте детальный анализ "
                "структуры, паттернов и зависимостей."
            ),
            name="code_analyzer_agent"
        )

        # Агент генерации кода
        def generate_code_with_context(requirements: str) -> str:
            """Генерация кода с учетом контекста проекта"""
            # Поиск похожих реализаций
            context_results = self.vector_store.search(requirements, k=3)

            context_info = "Контекст из проекта:\n"

            if context_results['documents']:
                for doc, meta in zip(
                    context_results['documents'], 
                    context_results['metadatas']
                ):
                    context_info += f"\nПример из {meta.get('file_path')}:\n"
                    context_info += f"{doc[:200]}...\n"

            return f"Генерация кода для: {requirements}\n\n{context_info}"

        self.agents['code_generator'] = create_react_agent(
            model=self.config.llm_model,
            tools=[generate_code_with_context],
            prompt=(
                "Вы - эксперт по генерации качественного кода. "
                "Используйте контекст проекта для создания кода, который "
                "соответствует стилю и архитектуре существующего проекта. "
                "Следуйте best practices и conventions проекта."
            ),
            name="code_generator_agent"
        )

        # Остальные агенты...
        self._create_specialized_agents()

        logger.info(f"Инициализировано {len(self.agents)} агентов")

    def _create_specialized_agents(self):
        """Создание остальных специализированных агентов"""

        # Агент тестирования
        def create_comprehensive_tests(code_input: str) -> str:
            """Создание comprehensive тестов"""
            return f"Создание тестов для:\n{code_input[:100]}...\n\nВключает: unit tests, integration tests, edge cases"

        self.agents['test_specialist'] = create_react_agent(
            model=self.config.llm_model,
            tools=[create_comprehensive_tests],
            prompt=(
                "Вы - эксперт по тестированию ПО. Создавайте comprehensive "
                "test suites с покрытием edge cases, mock объектов и "
                "integration тестов. Следуйте TDD принципам."
            ),
            name="test_specialist_agent"
        )

        # Агент документации
        def generate_technical_docs(code_or_api: str) -> str:
            """Генерация технической документации"""
            return f"Создание документации для:\n{code_or_api[:100]}...\n\nВключает: API docs, usage examples, architecture overview"

        self.agents['documentation_expert'] = create_react_agent(
            model=self.config.llm_model,
            tools=[generate_technical_docs],
            prompt=(
                "Вы - эксперт по технической документации. Создавайте "
                "clear, comprehensive документацию с примерами использования, "
                "API спецификациями и архитектурными диаграммами."
            ),
            name="documentation_expert_agent"
        )

        # Агент рефакторинга
        def refactor_and_optimize(legacy_code: str) -> str:
            """Рефакторинг и оптимизация кода"""
            # Поиск лучших практик в проекте
            best_practices = self.vector_store.search(f"best practices {legacy_code[:50]}", k=2)

            return f"Рефакторинг кода:\n{legacy_code[:150]}...\n\nПредложения по улучшению основаны на лучших практиках проекта"

        self.agents['refactoring_specialist'] = create_react_agent(
            model=self.config.llm_model,
            tools=[refactor_and_optimize],
            prompt=(
                "Вы - эксперт по рефакторингу и оптимизации кода. "
                "Улучшайте читаемость, производительность, maintainability "
                "и следование принципам SOLID. Предлагайте конкретные улучшения."
            ),
            name="refactoring_specialist_agent"
        )

    def _init_supervisor(self):
        """Инициализация супервизора"""
        self.supervisor = create_supervisor(
            agents=list(self.agents.values()),
            model=init_chat_model(self.config.llm_model),
            prompt=(
                "Вы - ведущий AI архитектор и ментор разработчиков. "
                "Управляйте командой экспертов:\n"
                "• code_analyzer_agent: анализ архитектуры и паттернов\n"
                "• code_generator_agent: генерация нового кода\n"
                "• test_specialist_agent: создание тестов\n"
                "• documentation_expert_agent: техническая документация\n"
                "• refactoring_specialist_agent: улучшение существующего кода\n\n"

                "Анализируйте запросы и делегируйте задачи подходящим экспертам. "
                "Для комплексных задач используйте несколько агентов последовательно. "
                "Всегда предоставляйте comprehensive ответы с примерами и объяснениями."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history"
        ).compile()

        logger.info("Супервизор инициализирован")

    async def index_project(self):
        """Асинхронная индексация проекта"""
        logger.info(f"Начало индексации проекта: {self.config.project_path}")

        all_documents = []

        # Сканирование файлов проекта
        project_path = Path(self.config.project_path)
        code_files = list(project_path.rglob("*.py"))  # Можно расширить для других языков

        logger.info(f"Найдено {len(code_files)} файлов для анализа")

        for file_path in code_files:
            try:
                chunks = self.code_analyzer.parse_file(str(file_path))
                all_documents.extend(chunks)

                if len(all_documents) % 100 == 0:
                    logger.info(f"Обработано {len(all_documents)} чанков кода")

            except Exception as e:
                logger.error(f"Ошибка обработки {file_path}: {e}")

        # Добавление в векторное хранилище
        if all_documents:
            self.vector_store.add_documents(all_documents)
            logger.info(f"Индексация завершена. Всего чанков: {len(all_documents)}")
        else:
            logger.warning("Не найдено документов для индексации")

    def process_query(self, user_query: str) -> str:
        """Обработка пользовательского запроса"""
        try:
            config = {"configurable": {"thread_id": "main_session"}}

            result = self.supervisor.invoke(
                {"messages": [{"role": "user", "content": user_query}]},
                config=config
            )

            return result["messages"][-1].content

        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return f"Извините, произошла ошибка: {str(e)}"

# ИНТЕРФЕЙС КОМАНДНОЙ СТРОКИ

class CodeAssistantCLI:
    """Интерфейс командной строки для AI помощника"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.agent_manager = AgentManager(self.config)
        self.session_history = []

    def _load_config(self, config_path: str) -> SystemConfig:
        """Загрузка конфигурации"""
        # Упрощенная загрузка конфигурации
        return SystemConfig(
            project_path="./",  # Замените на ваш путь
            vector_db_path="./code_embeddings",
            llm_model="ollama:qwen2.5-coder:7b"
        )

    async def initialize(self):
        """Инициализация системы"""
        print("🚀 Инициализация AI помощника разработчика...")

        # Проверка и индексация проекта
        if not Path(self.config.vector_db_path).exists():
            print("📚 Первичная индексация проекта...")
            await self.agent_manager.index_project()
        else:
            print("📚 Использование существующего индекса")

        print("✅ Система готова к работе!")

    def run_interactive(self):
        """Интерактивный режим работы"""
        print("\n" + "="*60)
        print("🤖 AI ПОМОЩНИК РАЗРАБОТЧИКА")
        print("="*60)
        print("Команды:")
        print("  • Анализ кода: 'найди функции для работы с API'")
        print("  • Генерация: 'создай класс для User management'") 
        print("  • Тестирование: 'напиши тесты для UserService'")
        print("  • Документация: 'создай README для этого проекта'")
        print("  • Рефакторинг: 'улучши этот код: [код]'")
        print("  • Выход: 'exit'")
        print("="*60)

        while True:
            try:
                user_input = input("\n📝 Ваш запрос: ").strip()

                if user_input.lower() in ['exit', 'quit', 'выход']:
                    print("\n👋 До свидания!")
                    break

                if not user_input:
                    continue

                print("\n🔄 Анализ запроса и делегирование экспертам...")

                # Обработка запроса
                response = self.agent_manager.process_query(user_input)

                print(f"\n✅ Ответ:\n{response}")

                # Сохранение в историю
                self.session_history.append({
                    'query': user_input,
                    'response': response
                })

            except KeyboardInterrupt:
                print("\n\n👋 Выход по Ctrl+C")
                break
            except Exception as e:
                print(f"\n❌ Произошла ошибка: {str(e)}")
                print("Попробуйте еще раз или используйте 'exit' для выхода")

# ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ

class ProjectAnalytics:
    """Аналитика проекта и метрики"""

    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def get_project_overview(self) -> Dict:
        """Получение обзора проекта"""
        # Получение статистики из векторной базы
        collection_info = self.vector_store.collection.count()

        return {
            'total_code_chunks': collection_info,
            'languages_detected': ['Python'],  # Можно расширить
            'complexity_distribution': {
                'low': 0, 'medium': 0, 'high': 0  # Можно вычислить реально
            }
        }

    def suggest_improvements(self) -> List[str]:
        """Предложения по улучшению проекта"""
        return [
            "Добавить больше unit тестов",
            "Улучшить документацию API",
            "Рефакторинг сложных функций",
            "Добавить type hints"
        ]

# ГЛАВНАЯ ФУНКЦИЯ

async def main():
    """Главная функция для запуска системы"""

    # Создание CLI интерфейса
    cli = CodeAssistantCLI()

    # Инициализация системы
    await cli.initialize()

    # Запуск интерактивного режима
    cli.run_interactive()

if __name__ == "__main__":
    # Запуск системы
    asyncio.run(main())
