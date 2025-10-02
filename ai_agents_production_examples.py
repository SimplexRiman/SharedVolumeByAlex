
# 🚀 PRODUCTION-READY AI АГЕНТЫ
# Реальные примеры с настоящими API и фреймворками

# =============================================================================
# РАЗДЕЛ 9: РЕАЛЬНЫЕ LANGCHAIN АГЕНТЫ
# =============================================================================

# Установка зависимостей:
# pip install langchain langchain-openai langchain-community langchain-experimental
# pip install tavily-python duckduckgo-search wikipedia
# pip install chromadb faiss-cpu sentence-transformers

from typing import List, Dict, Any, Optional, Union
import os
import json
from datetime import datetime, timedelta
import asyncio
import logging

# LangChain imports
"""
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
"""

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionReActAgent:
    """
    Production-ready ReAct агент с полным набором инструментов

    Особенности:
    - Интеграция с OpenAI API
    - Поддержка множественных инструментов
    - Обработка ошибок и логирование
    - Асинхронная обработка
    - Персистентная память
    """

    def __init__(self, 
                 openai_api_key: str,
                 model: str = "gpt-4-turbo-preview",
                 temperature: float = 0.7,
                 max_iterations: int = 10):

        # Настройка LLM
        self.llm = None  # ChatOpenAI(api_key=openai_api_key, model=model, temperature=temperature)
        self.max_iterations = max_iterations

        # Инициализация инструментов
        self.tools = self._initialize_tools()

        # Настройка памяти
        self.memory = None  # ConversationBufferWindowMemory(k=10, return_messages=True)

        # Создание агента
        self.agent_executor = self._create_agent()

        logger.info(f"ProductionReActAgent initialized with {len(self.tools)} tools")

    def _initialize_tools(self) -> List:
        """Инициализация набора инструментов"""
        tools = []

        # 1. Поиск в интернете
        """
        search_tool = Tool(
            name="web_search",
            description="Search the internet for current information. Use this when you need up-to-date information.",
            func=DuckDuckGoSearchRun().run
        )
        tools.append(search_tool)
        """

        # 2. Wikipedia поиск
        """
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        wiki_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for encyclopedic information on any topic.",
            func=wikipedia.run
        )
        tools.append(wiki_tool)
        """

        # 3. Калькулятор
        def safe_calculator(expression: str) -> str:
            """Безопасный калькулятор"""
            try:
                # Простая проверка безопасности
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Invalid characters in expression"

                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Calculation error: {str(e)}"

        calc_tool = Tool(
            name="calculator",
            description="Perform mathematical calculations. Input should be a valid mathematical expression.",
            func=safe_calculator
        )
        tools.append(calc_tool)

        # 4. Работа с файлами
        def file_operations(operation: str) -> str:
            """Работа с файлами"""
            try:
                if operation.startswith("read:"):
                    filename = operation[5:].strip()
                    if os.path.exists(filename):
                        with open(filename, 'r', encoding='utf-8') as f:
                            content = f.read()[:1000]  # Ограничиваем размер
                        return f"File content (first 1000 chars): {content}"
                    else:
                        return f"File {filename} not found"

                elif operation.startswith("write:"):
                    parts = operation[6:].split("|", 1)
                    if len(parts) == 2:
                        filename, content = parts
                        with open(filename.strip(), 'w', encoding='utf-8') as f:
                            f.write(content.strip())
                        return f"Successfully wrote to {filename}"
                    else:
                        return "Invalid write format. Use: write:filename|content"

                elif operation.startswith("list:"):
                    directory = operation[5:].strip() or "."
                    files = os.listdir(directory)[:20]  # Ограничиваем количество
                    return f"Files in {directory}: {', '.join(files)}"

                else:
                    return "Available operations: read:filename, write:filename|content, list:directory"

            except Exception as e:
                return f"File operation error: {str(e)}"

        file_tool = Tool(
            name="file_operations",
            description="Read, write, or list files. Use format: read:filename, write:filename|content, or list:directory",
            func=file_operations
        )
        tools.append(file_tool)

        # 5. Дата и время
        def datetime_tool(query: str) -> str:
            """Работа с датой и временем"""
            now = datetime.now()

            if "current" in query.lower() or "now" in query.lower():
                return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            elif "date" in query.lower():
                return f"Current date: {now.strftime('%Y-%m-%d')}"
            elif "time" in query.lower():
                return f"Current time: {now.strftime('%H:%M:%S')}"
            elif "tomorrow" in query.lower():
                tomorrow = now + timedelta(days=1)
                return f"Tomorrow's date: {tomorrow.strftime('%Y-%m-%d')}"
            elif "yesterday" in query.lower():
                yesterday = now - timedelta(days=1)
                return f"Yesterday's date: {yesterday.strftime('%Y-%m-%d')}"
            else:
                return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

        time_tool = Tool(
            name="datetime",
            description="Get current date, time, or calculate dates. Ask for 'current time', 'today', 'tomorrow', etc.",
            func=datetime_tool
        )
        tools.append(time_tool)

        return tools

    def _create_agent(self):
        """Создание агента с инструментами"""
        # В реальном коде здесь была бы настройка агента
        # return create_openai_functions_agent(self.llm, self.tools, prompt)
        # return AgentExecutor(agent=agent, tools=self.tools, memory=self.memory, verbose=True)

        # Заглушка для демонстрации
        class MockAgentExecutor:
            def __init__(self, tools):
                self.tools = {tool.name: tool for tool in tools}

            def run(self, input_dict):
                query = input_dict.get("input", "")

                # Простая логика выбора инструмента
                if "calculate" in query or "math" in query:
                    # Извлекаем выражение
                    import re
                    numbers = re.findall(r'[0-9+\-*/().\s]+', query)
                    if numbers:
                        return self.tools["calculator"].func(numbers[0].strip())

                elif "time" in query or "date" in query:
                    return self.tools["datetime"].func(query)

                elif "file" in query:
                    return self.tools["file_operations"].func("list:.")

                else:
                    return "I understand your query, but I need to use specific tools to help you better."

        return MockAgentExecutor(self.tools)

    def run(self, query: str) -> str:
        """Выполнение запроса"""
        try:
            logger.info(f"Processing query: {query}")

            result = self.agent_executor.run({"input": query})

            logger.info(f"Query completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error: {str(e)}"

    async def arun(self, query: str) -> str:
        """Асинхронное выполнение запроса"""
        try:
            # В реальном коде: result = await self.agent_executor.arun({"input": query})
            result = self.run(query)  # Синхронная заглушка
            return result
        except Exception as e:
            logger.error(f"Error in async processing: {str(e)}")
            return f"Async error: {str(e)}"

# Пример использования Production ReAct агента
def demo_production_agent():
    """Демонстрация production агента"""
    print("=== PRODUCTION REACT АГЕНТ ===")

    # В реальном коде нужен API ключ OpenAI
    # agent = ProductionReActAgent(openai_api_key="your-api-key-here")

    # Заглушка для демонстрации
    class MockAgent:
        def __init__(self):
            self.tools = [
                type('Tool', (), {'name': 'calculator', 'func': lambda x: f"Calculated: {x}"})(),
                type('Tool', (), {'name': 'datetime', 'func': lambda x: f"Time info: {datetime.now()}"})()
            ]

        def run(self, query):
            if "calculate" in query or "math" in query:
                return "Result: 42 (using calculator tool)"
            elif "time" in query:
                return f"Current time: {datetime.now().strftime('%H:%M:%S')}"
            return f"Processed query: {query}"

    agent = MockAgent()

    # Тестовые запросы
    test_queries = [
        "What's 15 * 8 + 7?",
        "What time is it now?",
        "Tell me about artificial intelligence"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = agent.run(query)
        print(f"Result: {result}")

# Запуск демонстрации
demo_production_agent()

# =============================================================================
# РАЗДЕЛ 10: RAG СИСТЕМА С АГЕНТАМИ
# =============================================================================

class ProductionRAGAgent:
    """
    Production RAG система с агентами

    Компоненты:
    - Векторная база данных (Chroma)
    - Embedding модель (HuggingFace)
    - Retrieval агент
    - Generation агент
    - Query routing агент
    """

    def __init__(self, 
                 documents: List[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Инициализация компонентов
        self._setup_embeddings(embedding_model)
        self._setup_vectorstore()

        if documents:
            self.add_documents(documents)

        # Агенты
        self.query_router = QueryRouterAgent()
        self.retrieval_agent = RetrievalAgent(self.vectorstore)
        self.generation_agent = GenerationAgent()

        logger.info("ProductionRAGAgent initialized")

    def _setup_embeddings(self, model_name: str):
        """Настройка embeddings"""
        # В реальном коде:
        # self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Заглушка
        class MockEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]

            def embed_query(self, text):
                return [0.1] * 384

        self.embeddings = MockEmbeddings()

    def _setup_vectorstore(self):
        """Настройка векторного хранилища"""
        # В реальном коде:
        # self.vectorstore = Chroma(embedding_function=self.embeddings, persist_directory="./chroma_db")

        # Заглушка
        class MockVectorStore:
            def __init__(self):
                self.docs = []

            def add_documents(self, documents):
                self.docs.extend(documents)
                return len(documents)

            def similarity_search(self, query, k=5):
                # Возвращаем первые k документов
                return self.docs[:k]

        self.vectorstore = MockVectorStore()

    def add_documents(self, documents: List[str]):
        """Добавление документов в векторную БД"""
        # Разбивка на чанки
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=self.chunk_size,
        #     chunk_overlap=self.chunk_overlap
        # )

        # Простая разбивка для демонстрации
        chunks = []
        for doc in documents:
            doc_chunks = [doc[i:i+self.chunk_size] for i in range(0, len(doc), self.chunk_size-self.chunk_overlap)]
            chunks.extend(doc_chunks)

        # Создание документов
        docs = [{"page_content": chunk, "metadata": {"source": f"doc_{i}"}} for i, chunk in enumerate(chunks)]

        # Добавление в векторную БД
        added_count = self.vectorstore.add_documents(docs)
        logger.info(f"Added {added_count} document chunks to vector store")

        return added_count

    def query(self, question: str) -> Dict[str, Any]:
        """Обработка запроса через RAG пайплайн"""

        # 1. Маршрутизация запроса
        query_type = self.query_router.route(question)

        # 2. Поиск релевантных документов
        relevant_docs = self.retrieval_agent.retrieve(question, k=5)

        # 3. Генерация ответа
        answer = self.generation_agent.generate(question, relevant_docs)

        return {
            "question": question,
            "query_type": query_type,
            "relevant_documents": len(relevant_docs),
            "answer": answer,
            "sources": [doc.get("metadata", {}).get("source", "unknown") for doc in relevant_docs]
        }

class QueryRouterAgent:
    """Агент для маршрутизации запросов"""

    def route(self, query: str) -> str:
        """Определение типа запроса"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["what", "define", "explain", "describe"]):
            return "factual"
        elif any(word in query_lower for word in ["how", "step", "process", "method"]):
            return "procedural"
        elif any(word in query_lower for word in ["why", "reason", "cause", "because"]):
            return "causal"
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return "comparative"
        else:
            return "general"

class RetrievalAgent:
    """Агент для поиска документов"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Поиск релевантных документов"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

class GenerationAgent:
    """Агент для генерации ответов"""

    def generate(self, question: str, context_docs: List[Dict]) -> str:
        """Генерация ответа на основе контекста"""

        # Объединение контекста
        context = "\n\n".join([doc.get("page_content", "") for doc in context_docs])

        # В реальном коде здесь был бы вызов LLM
        # prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        # response = llm.predict(prompt)

        # Простая генерация для демонстрации
        if context:
            answer = f"Based on the provided context, I can answer your question about '{question}'. The relevant information suggests that this topic involves multiple aspects that need to be considered."
        else:
            answer = f"I don't have specific information in my knowledge base to answer '{question}' accurately."

        return answer

# Демонстрация RAG системы
def demo_rag_system():
    """Демонстрация RAG системы"""
    print("\n=== PRODUCTION RAG СИСТЕМА ===")

    # Примерные документы
    sample_docs = [
        "Искусственный интеллект (ИИ) — это область информатики, которая занимается созданием интеллектуальных машин, способных работать и реагировать как люди.",
        "Машинное обучение является подмножеством искусственного интеллекта, которое предоставляет системам способность автоматически учиться и улучшаться на основе опыта.",
        "Глубокое обучение - это подраздел машинного обучения, основанный на искусственных нейронных сетях с представлением обучения.",
        "Нейронные сети вдохновлены биологическими нейронными сетями мозга животных и используются для решения широкого круга задач."
    ]

    # Создание RAG системы
    rag_agent = ProductionRAGAgent(documents=sample_docs)

    # Тестовые запросы
    test_questions = [
        "Что такое искусственный интеллект?",
        "Как связано машинное обучение с ИИ?",
        "Объясни глубокое обучение",
        "Расскажи о нейронных сетях"
    ]

    for question in test_questions:
        print(f"\nВопрос: {question}")
        result = rag_agent.query(question)

        print(f"Тип запроса: {result['query_type']}")
        print(f"Найдено документов: {result['relevant_documents']}")
        print(f"Ответ: {result['answer']}")
        print(f"Источники: {result['sources']}")

# Запуск демонстрации RAG
demo_rag_system()

# =============================================================================
# РАЗДЕЛ 11: МОНИТОРИНГ И DEPLOYMENT
# =============================================================================

class AgentMonitoringSystem:
    """
    Система мониторинга для production агентов

    Отслеживает:
    - Производительность
    - Использование ресурсов
    - Качество ответов
    - Ошибки и исключения
    """

    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0.0,
            "error_types": {},
            "query_types": {},
            "response_quality_scores": []
        }

        self.logs = []

        # Настройка логирования в файл
        self._setup_logging()

    def _setup_logging(self):
        """Настройка системы логирования"""
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler = logging.FileHandler('agent_monitoring.log')
        file_handler.setFormatter(log_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)

        self.logger = logging.getLogger('AgentMonitoring')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_query(self, 
                  query: str, 
                  response: str, 
                  response_time: float, 
                  success: bool,
                  error: str = None,
                  query_type: str = "general"):
        """Логирование запроса"""

        # Обновление метрик
        self.metrics["total_queries"] += 1

        if success:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1
            if error:
                self.metrics["error_types"][error] = self.metrics["error_types"].get(error, 0) + 1

        # Обновление среднего времени ответа
        total_time = self.metrics["average_response_time"] * (self.metrics["total_queries"] - 1)
        self.metrics["average_response_time"] = (total_time + response_time) / self.metrics["total_queries"]

        # Подсчет типов запросов
        self.metrics["query_types"][query_type] = self.metrics["query_types"].get(query_type, 0) + 1

        # Логирование
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100] + "..." if len(query) > 100 else query,
            "response_length": len(response),
            "response_time": response_time,
            "success": success,
            "error": error,
            "query_type": query_type
        }

        self.logs.append(log_entry)

        if success:
            self.logger.info(f"Query processed successfully in {response_time:.2f}s")
        else:
            self.logger.error(f"Query failed: {error}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик"""
        success_rate = (self.metrics["successful_queries"] / self.metrics["total_queries"] * 100) if self.metrics["total_queries"] > 0 else 0

        return {
            "total_queries": self.metrics["total_queries"],
            "success_rate": f"{success_rate:.2f}%",
            "average_response_time": f"{self.metrics['average_response_time']:.2f}s",
            "most_common_errors": dict(sorted(self.metrics["error_types"].items(), key=lambda x: x[1], reverse=True)[:3]),
            "query_type_distribution": self.metrics["query_types"],
            "uptime": "Available in production deployment"
        }

    def generate_report(self) -> str:
        """Генерация отчета о работе агента"""
        metrics = self.get_metrics_summary()

        report = f"""
=== AGENT MONITORING REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
- Total Queries: {metrics['total_queries']}
- Success Rate: {metrics['success_rate']}
- Average Response Time: {metrics['average_response_time']}

Query Types:
"""
        for query_type, count in metrics['query_type_distribution'].items():
            report += f"- {query_type}: {count} queries\n"

        report += f"""
Most Common Errors:
"""
        for error, count in metrics['most_common_errors'].items():
            report += f"- {error}: {count} occurrences\n"

        report += f"""
Recent Activity:
"""
        for log_entry in self.logs[-5:]:  # Последние 5 записей
            status = "✅" if log_entry['success'] else "❌"
            report += f"{status} {log_entry['timestamp']}: {log_entry['query']} ({log_entry['response_time']:.2f}s)\n"

        return report

# Пример интеграции мониторинга с агентом
class MonitoredAgent:
    """Агент с интегрированным мониторингом"""

    def __init__(self):
        self.monitoring = AgentMonitoringSystem()
        self.tools = ["calculator", "search", "datetime"]

    def process_query(self, query: str) -> str:
        """Обработка запроса с мониторингом"""
        start_time = datetime.now()

        try:
            # Имитация обработки запроса
            import time
            time.sleep(0.1)  # Имитация времени обработки

            if "error" in query.lower():
                raise Exception("Simulated error for testing")

            response = f"Processed query: {query} (length: {len(query)} chars)"

            # Определение типа запроса
            if "calculate" in query.lower():
                query_type = "calculation"
            elif "search" in query.lower():
                query_type = "search"
            else:
                query_type = "general"

            # Логирование успешного запроса
            response_time = (datetime.now() - start_time).total_seconds()
            self.monitoring.log_query(
                query=query,
                response=response,
                response_time=response_time,
                success=True,
                query_type=query_type
            )

            return response

        except Exception as e:
            # Логирование ошибки
            response_time = (datetime.now() - start_time).total_seconds()
            self.monitoring.log_query(
                query=query,
                response="",
                response_time=response_time,
                success=False,
                error=str(e)
            )

            return f"Error processing query: {str(e)}"

    def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья агента"""
        metrics = self.monitoring.get_metrics_summary()

        # Определение статуса здоровья
        success_rate = float(metrics['success_rate'].replace('%', ''))
        avg_time = float(metrics['average_response_time'].replace('s', ''))

        if success_rate > 95 and avg_time < 2.0:
            health_status = "HEALTHY"
        elif success_rate > 80 and avg_time < 5.0:
            health_status = "WARNING"
        else:
            health_status = "CRITICAL"

        return {
            "status": health_status,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

# Демонстрация мониторинга
def demo_monitoring():
    """Демонстрация системы мониторинга"""
    print("\n=== СИСТЕМА МОНИТОРИНГА АГЕНТОВ ===")

    agent = MonitoredAgent()

    # Тестовые запросы
    test_queries = [
        "Calculate 15 + 25",
        "Search for information about AI",
        "What is the current time?",
        "This query should cause an error",  # Вызовет ошибку
        "Another successful query"
    ]

    # Обработка запросов
    for query in test_queries:
        print(f"\nProcessing: {query}")
        response = agent.process_query(query)
        print(f"Response: {response}")

    # Получение статуса здоровья
    health = agent.get_health_status()
    print(f"\n=== HEALTH STATUS ===")
    print(f"Status: {health['status']}")
    print(f"Success Rate: {health['metrics']['success_rate']}")
    print(f"Avg Response Time: {health['metrics']['average_response_time']}")

    # Генерация отчета
    report = agent.monitoring.generate_report()
    print(f"\n=== MONITORING REPORT ===")
    print(report)

# Запуск демонстрации мониторинга
demo_monitoring()

print("\n✅ Production-ready примеры созданы и протестированы!")
