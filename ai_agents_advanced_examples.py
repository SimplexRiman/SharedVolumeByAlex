
# 🚀 ПРОДВИНУТЫЕ AI АГЕНТЫ - СОВРЕМЕННЫЕ ФРЕЙМВОРКИ
# LangChain, LangGraph, AutoGen, CrewAI, Production Examples

# =============================================================================
# РАЗДЕЛ 6: LANGCHAIN АГЕНТЫ
# =============================================================================

# 6.1 Базовый LangChain ReAct агент
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Имитация LangChain компонентов для демонстрации структуры
class MockChatOpenAI:
    """Имитация ChatOpenAI для демонстрации"""
    def __init__(self, model="gpt-4", temperature=0.7):
        self.model = model
        self.temperature = temperature

    def predict(self, text: str) -> str:
        # Простая имитация ответов LLM
        if "weather" in text.lower():
            return "I need to search for weather information. Action: search_weather[current weather]"
        elif "calculate" in text.lower() or "math" in text.lower():
            return "I need to perform a calculation. Action: calculator[2+2]"
        else:
            return "I can answer this directly based on my knowledge."

class BaseTool:
    """Базовый класс инструмента"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, input_str: str) -> str:
        raise NotImplementedError

class CalculatorTool(BaseTool):
    """Инструмент калькулятора"""
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Useful for mathematical calculations. Input should be a math expression."
        )

    def run(self, input_str: str) -> str:
        try:
            result = eval(input_str)  # В продакшене использовать безопасный парсер
            return f"The result is: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"

class SearchTool(BaseTool):
    """Инструмент поиска"""
    def __init__(self):
        super().__init__(
            name="search",
            description="Useful for searching current information on the internet."
        )

    def run(self, input_str: str) -> str:
        # Имитация поиска
        fake_results = {
            "weather": "Current weather: Sunny, 22°C in Moscow",
            "news": "Latest news: AI agents are transforming industries",
            "python": "Python is a high-level programming language"
        }

        for key in fake_results:
            if key in input_str.lower():
                return fake_results[key]

        return f"Search results for '{input_str}': Information found."

class LangChainReActAgent:
    """
    LangChain-style ReAct агент
    Демонстрирует архитектуру ReAct с инструментами
    """

    def __init__(self, llm, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = 5

    def _format_tools(self) -> str:
        """Форматирование описаний инструментов"""
        tool_descriptions = []
        for tool in self.tools.values():
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descriptions)

    def _create_react_prompt(self, query: str, scratchpad: str = "") -> str:
        """Создание промпта в стиле ReAct"""
        tools_text = self._format_tools()

        prompt = f"""Answer the following questions as best you can. You have access to the following tools:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{', '.join(self.tools.keys())}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{scratchpad}"""
        return prompt

    def _parse_action(self, text: str) -> tuple:
        """Парсинг действия из ответа LLM"""
        lines = text.split('\n')
        action = None
        action_input = None

        for line in lines:
            if line.startswith("Action:"):
                action = line.split("Action:")[1].strip()
            elif line.startswith("Action Input:"):
                action_input = line.split("Action Input:")[1].strip()

        return action, action_input

    def run(self, query: str) -> str:
        """Выполнение ReAct цикла"""
        scratchpad = "Thought: I need to understand what the user is asking for."

        for iteration in range(self.max_iterations):
            prompt = self._create_react_prompt(query, scratchpad)

            # Получаем ответ от LLM
            llm_output = self.llm.predict(prompt)
            print(f"\nИтерация {iteration + 1}:")
            print(f"LLM Output: {llm_output}")

            # Если есть финальный ответ
            if "Final Answer:" in llm_output:
                final_answer = llm_output.split("Final Answer:")[1].strip()
                return final_answer

            # Парсим действие
            action, action_input = self._parse_action(llm_output)

            if action and action in self.tools:
                # Выполняем инструмент
                tool = self.tools[action]
                observation = tool.run(action_input)

                print(f"Action: {action}")
                print(f"Action Input: {action_input}")
                print(f"Observation: {observation}")

                # Обновляем scratchpad
                scratchpad += f"\nThought: {llm_output}\n"
                scratchpad += f"Action: {action}\n"
                scratchpad += f"Action Input: {action_input}\n"
                scratchpad += f"Observation: {observation}\n"
                scratchpad += f"Thought:"
            else:
                # Нет валидного действия
                return llm_output

        return "Maximum iterations reached"

# Пример использования LangChain ReAct агента
print("=== LANGCHAIN REACT АГЕНТ ===")

llm = MockChatOpenAI()
tools = [CalculatorTool(), SearchTool()]
agent = LangChainReActAgent(llm, tools)

result = agent.run("What's 25 * 4 + 10?")
print(f"\nФинальный результат: {result}")

# 6.2 LangChain агент с памятью
class ConversationMemory:
    """Простая реализация памяти разговора"""

    def __init__(self, max_length: int = 10):
        self.messages = []
        self.max_length = max_length

    def add_message(self, role: str, content: str):
        """Добавление сообщения в память"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

        # Ограничение размера памяти
        if len(self.messages) > self.max_length:
            self.messages = self.messages[-self.max_length:]

    def get_context(self) -> str:
        """Получение контекста разговора"""
        context_lines = []
        for msg in self.messages[-6:]:  # Последние 6 сообщений
            context_lines.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(context_lines)

class MemoryEnhancedAgent:
    """
    Агент с долгосрочной памятью
    Помнит предыдущие взаимодействия и использует контекст
    """

    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationMemory()
        self.user_profile = {}

    def _create_contextualized_prompt(self, query: str) -> str:
        """Создание промпта с контекстом"""
        context = self.memory.get_context()

        prompt = f"""You are a helpful AI assistant with memory of our conversation.

Previous conversation context:
{context}

Current user query: {query}

Please provide a helpful response that takes into account our conversation history.
"""
        return prompt

    def chat(self, user_input: str) -> str:
        """Чат с памятью"""
        # Добавляем пользовательский ввод в память
        self.memory.add_message("User", user_input)

        # Создаем промпт с контекстом
        prompt = self._create_contextualized_prompt(user_input)

        # Получаем ответ
        response = self.llm.predict(prompt)

        # Добавляем ответ в память
        self.memory.add_message("Assistant", response)

        return response

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Получение сводки разговора"""
        return {
            "total_messages": len(self.memory.messages),
            "recent_topics": [msg["content"][:50] + "..." for msg in self.memory.messages[-3:]],
            "user_profile": self.user_profile
        }

# Пример использования агента с памятью
print("\n=== АГЕНТ С ПАМЯТЬЮ ===")

memory_agent = MemoryEnhancedAgent(MockChatOpenAI())

# Симуляция диалога
responses = []
queries = [
    "Меня зовут Алексей, я программист",
    "Какие языки программирования популярны сейчас?", 
    "А что насчет Python?",
    "Помнишь, как меня зовут?"
]

for query in queries:
    response = memory_agent.chat(query)
    responses.append(f"User: {query}\nBot: {response}\n")
    print(f"User: {query}")
    print(f"Bot: {response}\n")

summary = memory_agent.get_conversation_summary()
print(f"Сводка разговора: {summary}")

# =============================================================================
# РАЗДЕЛ 7: LANGGRAPH МУЛЬТИ-АГЕНТНАЯ СИСТЕМА
# =============================================================================

class AgentState:
    """Состояние агента в LangGraph-style системе"""

    def __init__(self):
        self.messages = []
        self.current_task = None
        self.results = {}
        self.next_agent = None

class LangGraphNode:
    """Базовый узел в LangGraph"""

    def __init__(self, name: str):
        self.name = name

    def process(self, state: AgentState) -> AgentState:
        """Обработка состояния узлом"""
        raise NotImplementedError

class SupervisorNode(LangGraphNode):
    """Узел-супервизор для координации агентов"""

    def __init__(self):
        super().__init__("supervisor")
        self.agents = ["researcher", "writer", "reviewer"]

    def process(self, state: AgentState) -> AgentState:
        """Определение следующего агента"""
        if not state.current_task:
            state.next_agent = "researcher"
            state.current_task = "research"
        elif "research" in state.results:
            if "draft" not in state.results:
                state.next_agent = "writer"
                state.current_task = "writing"
            elif "review" not in state.results:
                state.next_agent = "reviewer"
                state.current_task = "review"
            else:
                state.next_agent = "END"

        print(f"Supervisor: Next agent -> {state.next_agent}")
        return state

class ResearcherNode(LangGraphNode):
    """Узел-исследователь"""

    def __init__(self):
        super().__init__("researcher")

    def process(self, state: AgentState) -> AgentState:
        """Исследование темы"""
        if state.current_task == "research":
            # Имитация исследования
            research_result = {
                "topic": "AI Agents",
                "key_points": [
                    "AI agents are autonomous software entities",
                    "They can perceive, reason, and act",
                    "Popular frameworks include LangChain, AutoGen"
                ],
                "sources": ["arxiv.org", "openai.com", "langchain.dev"]
            }

            state.results["research"] = research_result
            state.messages.append(f"Research completed: {research_result['topic']}")
            print(f"Researcher: Completed research on {research_result['topic']}")

        return state

class WriterNode(LangGraphNode):
    """Узел-писатель"""

    def __init__(self):
        super().__init__("writer")

    def process(self, state: AgentState) -> AgentState:
        """Написание контента"""
        if state.current_task == "writing" and "research" in state.results:
            research = state.results["research"]

            draft = f"""# {research['topic']}

## Introduction
{research['topic']} представляют собой важную область искусственного интеллекта.

## Key Points
"""
            for point in research['key_points']:
                draft += f"- {point}\n"

            draft += f"""
## Sources
Based on research from: {', '.join(research['sources'])}
"""

            state.results["draft"] = draft
            state.messages.append("Draft article completed")
            print("Writer: Draft completed")

        return state

class ReviewerNode(LangGraphNode):
    """Узел-рецензент"""

    def __init__(self):
        super().__init__("reviewer")

    def process(self, state: AgentState) -> AgentState:
        """Проверка и улучшение контента"""
        if state.current_task == "review" and "draft" in state.results:
            draft = state.results["draft"]

            # Простая проверка и улучшение
            reviewed = draft.replace("## Introduction", "## Введение")
            reviewed = reviewed.replace("## Key Points", "## Ключевые моменты")
            reviewed += "\n## Заключение\nAI агенты будут играть ключевую роль в будущем ИИ."

            state.results["review"] = reviewed
            state.results["final_article"] = reviewed
            state.messages.append("Article reviewed and finalized")
            print("Reviewer: Review completed")

        return state

class LangGraphWorkflow:
    """
    Workflow в стиле LangGraph
    Координирует выполнение узлов на основе состояния
    """

    def __init__(self):
        self.nodes = {
            "supervisor": SupervisorNode(),
            "researcher": ResearcherNode(),
            "writer": WriterNode(),
            "reviewer": ReviewerNode()
        }

        self.edges = {
            "supervisor": ["researcher", "writer", "reviewer", "END"],
            "researcher": ["supervisor"],
            "writer": ["supervisor"], 
            "reviewer": ["supervisor"]
        }

    def run(self, initial_query: str) -> AgentState:
        """Запуск workflow"""
        state = AgentState()
        state.messages = [f"Initial query: {initial_query}"]

        current_node = "supervisor"
        max_iterations = 10

        print(f"\n=== LANGGRAPH WORKFLOW START ===")
        print(f"Query: {initial_query}")

        for iteration in range(max_iterations):
            if current_node == "END":
                print("\n=== WORKFLOW COMPLETED ===")
                break

            print(f"\n--- Iteration {iteration + 1}: {current_node} ---")

            # Обрабатываем текущий узел
            if current_node in self.nodes:
                node = self.nodes[current_node]
                state = node.process(state)

                # Определяем следующий узел
                if current_node == "supervisor":
                    current_node = state.next_agent
                else:
                    current_node = "supervisor"
            else:
                print(f"Unknown node: {current_node}")
                break

        return state

# Пример использования LangGraph workflow
print("\n=== LANGGRAPH МУЛЬТИ-АГЕНТНАЯ СИСТЕМА ===")

workflow = LangGraphWorkflow()
result_state = workflow.run("Write an article about AI Agents")

print(f"\n=== РЕЗУЛЬТАТЫ ===")
for key, value in result_state.results.items():
    print(f"\n{key.upper()}:")
    if isinstance(value, str):
        print(value[:200] + "..." if len(value) > 200 else value)
    else:
        print(value)

# =============================================================================
# РАЗДЕЛ 8: CREWAI СТИЛЬ КОМАНД
# =============================================================================

class CrewAIAgent:
    """Агент в стиле CrewAI с ролями и задачами"""

    def __init__(self, role: str, goal: str, backstory: str, tools: List[str] = None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.memory = []

    def execute_task(self, task: str) -> str:
        """Выполнение задачи агентом"""
        print(f"\n{self.role} executing: {task}")
        print(f"Goal: {self.goal}")

        # Имитация выполнения задачи на основе роли
        if "research" in task.lower() and "researcher" in self.role.lower():
            result = f"Research completed: Found 5 key sources on the topic. Main insights: AI agents are becoming mainstream in 2024."
        elif "write" in task.lower() and "writer" in self.role.lower():
            result = f"Article drafted: 1200 words covering key aspects of AI agents, their applications, and future trends."
        elif "review" in task.lower() and ("editor" in self.role.lower() or "reviewer" in self.role.lower()):
            result = f"Content reviewed: Fixed 3 grammatical errors, improved clarity in 2 sections, added conclusion paragraph."
        else:
            result = f"Task completed using {self.role} expertise"

        self.memory.append({"task": task, "result": result})
        return result

class CrewAITask:
    """Задача для команды CrewAI"""

    def __init__(self, description: str, agent: CrewAIAgent, expected_output: str = None):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.result = None

    def execute(self) -> str:
        """Выполнение задачи"""
        self.result = self.agent.execute_task(self.description)
        return self.result

class CrewAICrew:
    """Команда агентов CrewAI"""

    def __init__(self, agents: List[CrewAIAgent], tasks: List[CrewAITask]):
        self.agents = agents
        self.tasks = tasks
        self.results = []

    def kickoff(self) -> List[str]:
        """Запуск выполнения всех задач"""
        print("\n=== CREWAI CREW KICKOFF ===")

        for i, task in enumerate(self.tasks, 1):
            print(f"\n--- Task {i} ---")
            print(f"Description: {task.description}")

            result = task.execute()
            self.results.append(result)

            print(f"Result: {result}")

        return self.results

    def get_crew_summary(self) -> Dict[str, Any]:
        """Получение сводки по команде"""
        return {
            "total_agents": len(self.agents),
            "total_tasks": len(self.tasks),
            "completed_tasks": len([t for t in self.tasks if t.result]),
            "agents": [{"role": a.role, "tools": len(a.tools)} for a in self.agents]
        }

# Создание CrewAI команды
print("\n=== CREWAI КОМАНДА ===")

# Создаем агентов с ролями
researcher = CrewAIAgent(
    role="Senior Researcher",
    goal="Conduct thorough research on AI agents and provide comprehensive insights",
    backstory="You are an experienced researcher with 10 years in AI field. You excel at finding relevant information and synthesizing key insights.",
    tools=["search", "analysis", "documentation"]
)

writer = CrewAIAgent(
    role="Technical Writer", 
    goal="Create engaging and informative articles based on research",
    backstory="You are a skilled technical writer who can transform complex research into accessible content.",
    tools=["writing", "editing", "formatting"]
)

editor = CrewAIAgent(
    role="Content Editor",
    goal="Review and refine content for quality and clarity",
    backstory="You are a meticulous editor with an eye for detail and excellent command of language.",
    tools=["proofreading", "editing", "quality_assurance"]
)

# Создаем задачи
task1 = CrewAITask(
    description="Research the latest trends in AI agents for 2024",
    agent=researcher,
    expected_output="Comprehensive research report with key findings"
)

task2 = CrewAITask(
    description="Write an article based on the research findings",
    agent=writer,
    expected_output="Well-structured 1000+ word article"
)

task3 = CrewAITask(
    description="Review and edit the article for publication",
    agent=editor,
    expected_output="Polished, publication-ready article"
)

# Создаем и запускаем команду
crew = CrewAICrew(
    agents=[researcher, writer, editor],
    tasks=[task1, task2, task3]
)

results = crew.kickoff()

print(f"\n=== CREW SUMMARY ===")
summary = crew.get_crew_summary()
print(f"Team composition: {summary}")

print("\n✅ Все продвинутые примеры созданы!")
