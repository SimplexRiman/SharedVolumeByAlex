
# 🚀 ПОЛНЫЙ ГАЙД ПО СОЗДАНИЮ AI АГЕНТОВ
# От простых примеров до продвинутых архитектур

# =============================================================================
# РАЗДЕЛ 1: ПРОСТЫЕ АГЕНТЫ - ОСНОВЫ
# =============================================================================

# 1.1 Простейший агент с правилами (Rule-Based Agent)
class SimpleReflexAgent:
    """
    Простейший рефлексный агент, реагирующий на условия
    Принцип: условие -> действие (без памяти состояния)
    """

    def __init__(self):
        self.rules = {
            "холодно": "включить_отопление",
            "жарко": "включить_кондиционер", 
            "темно": "включить_свет",
            "светло": "выключить_свет"
        }

    def perceive(self, environment):
        """Восприятие окружающей среды"""
        return environment.get("temperature", "normal")

    def act(self, condition):
        """Действие на основе правил"""
        action = self.rules.get(condition, "ничего_не_делать")
        print(f"Условие: {condition} -> Действие: {action}")
        return action

# Пример использования
environment = {"temperature": "холодно"}
agent = SimpleReflexAgent()
condition = agent.perceive(environment)
agent.act(condition)

# 1.2 Агент с внутренним состоянием (Model-Based Agent)
class ModelBasedAgent:
    """
    Агент с моделью мира и внутренним состоянием
    Принцип: состояние + восприятие -> новое состояние -> действие
    """

    def __init__(self):
        self.world_model = {
            "temperature": 20,
            "lights_on": False,
            "heater_on": False
        }
        self.goals = {"comfortable_temp": (18, 24)}

    def update_model(self, perception):
        """Обновление модели мира"""
        for key, value in perception.items():
            if key in self.world_model:
                self.world_model[key] = value

        print(f"Модель мира: {self.world_model}")

    def choose_action(self):
        """Выбор действия на основе состояния"""
        temp = self.world_model["temperature"]
        min_temp, max_temp = self.goals["comfortable_temp"]

        if temp < min_temp and not self.world_model["heater_on"]:
            return "включить_отопление"
        elif temp > max_temp and self.world_model["heater_on"]:
            return "выключить_отопление"
        else:
            return "поддерживать_текущее_состояние"

# Пример использования
agent = ModelBasedAgent()
perception = {"temperature": 15}
agent.update_model(perception)
action = agent.choose_action()
print(f"Действие: {action}")

# =============================================================================
# РАЗДЕЛ 2: АГЕНТЫ С ЦЕЛЯМИ И ПЛАНИРОВАНИЕМ
# =============================================================================

# 2.1 Целеориентированный агент (Goal-Based Agent)
import heapq
from typing import List, Dict, Tuple

class GoalBasedAgent:
    """
    Агент, планирующий действия для достижения целей
    Принцип: цель -> планирование -> выполнение плана
    """

    def __init__(self):
        self.state = {"location": "A", "energy": 100}
        self.goals = {"reach_location": "D"}
        self.actions = {
            "move_A_to_B": {"cost": 10, "effect": {"location": "B"}},
            "move_B_to_C": {"cost": 15, "effect": {"location": "C"}},
            "move_C_to_D": {"cost": 12, "effect": {"location": "D"}},
            "move_A_to_C": {"cost": 20, "effect": {"location": "C"}},
        }

    def is_goal_achieved(self):
        """Проверка достижения цели"""
        return self.state["location"] == self.goals["reach_location"]

    def find_path(self, start: str, goal: str) -> List[str]:
        """Простой поиск пути (A* упрощенный)"""
        # Упрощенная карта соединений
        graph = {
            "A": [("B", 10), ("C", 20)],
            "B": [("C", 15)], 
            "C": [("D", 12)]
        }

        # Эвристика (расстояние до цели)
        heuristic = {"A": 3, "B": 2, "C": 1, "D": 0}

        # Простая реализация A*
        queue = [(heuristic[start], 0, start, [start])]
        visited = set()

        while queue:
            f, g, current, path = heapq.heappop(queue)

            if current == goal:
                return path

            if current in visited:
                continue
            visited.add(current)

            if current in graph:
                for neighbor, cost in graph[current]:
                    if neighbor not in visited:
                        new_g = g + cost
                        new_f = new_g + heuristic[neighbor]
                        heapq.heappush(queue, (new_f, new_g, neighbor, path + [neighbor]))

        return []

    def execute_plan(self):
        """Выполнение плана для достижения цели"""
        if self.is_goal_achieved():
            print("Цель уже достигнута!")
            return

        path = self.find_path(self.state["location"], self.goals["reach_location"])
        print(f"Найден путь: {' -> '.join(path)}")

        # Выполнение действий
        for i in range(len(path) - 1):
            current = path[i]
            next_loc = path[i + 1]
            action_key = f"move_{current}_to_{next_loc}"

            if action_key in self.actions:
                action = self.actions[action_key]
                print(f"Выполняю: {action_key}")

                # Обновление состояния
                self.state.update(action["effect"])
                self.state["energy"] -= action["cost"]

                print(f"Новое состояние: {self.state}")

# Пример использования
agent = GoalBasedAgent()
agent.execute_plan()

# 2.2 Агент с утилитарной функцией (Utility-Based Agent)
class UtilityBasedAgent:
    """
    Агент, максимизирующий полезность действий
    Принцип: оценка полезности всех действий -> выбор наилучшего
    """

    def __init__(self):
        self.state = {
            "health": 80,
            "energy": 60, 
            "money": 100,
            "happiness": 70
        }

        self.actions = {
            "work": {
                "effects": {"energy": -20, "money": +50, "happiness": -10},
                "requirements": {"energy": 30}
            },
            "rest": {
                "effects": {"energy": +30, "health": +10, "happiness": +5},
                "requirements": {}
            },
            "exercise": {
                "effects": {"health": +20, "energy": -15, "happiness": +15},
                "requirements": {"energy": 20}
            },
            "socialize": {
                "effects": {"happiness": +25, "energy": -10, "money": -20},
                "requirements": {"money": 20}
            }
        }

    def calculate_utility(self, action_name: str) -> float:
        """Расчет полезности действия"""
        action = self.actions[action_name]

        # Проверка возможности выполнения
        for req, min_val in action["requirements"].items():
            if self.state[req] < min_val:
                return -float('inf')  # Невозможно выполнить

        # Расчет ожидаемой полезности
        utility = 0
        for attr, change in action["effects"].items():
            new_value = self.state[attr] + change

            # Функция полезности (предпочтение баланса)
            if attr == "health":
                utility += change * (1.0 if new_value <= 100 else 0.5)
            elif attr == "energy":
                utility += change * (1.2 if new_value <= 100 else 0.3)
            elif attr == "money":
                utility += change * 0.8
            elif attr == "happiness":
                utility += change * 1.5

        return utility

    def choose_best_action(self) -> str:
        """Выбор действия с максимальной полезностью"""
        best_action = None
        best_utility = -float('inf')

        print("Оценка действий:")
        for action_name in self.actions:
            utility = self.calculate_utility(action_name)
            print(f"  {action_name}: {utility:.2f}")

            if utility > best_utility:
                best_utility = utility
                best_action = action_name

        return best_action

    def execute_action(self, action_name: str):
        """Выполнение действия"""
        if action_name not in self.actions:
            return

        action = self.actions[action_name]
        print(f"Выполняю действие: {action_name}")
        print(f"Состояние до: {self.state}")

        # Применение эффектов
        for attr, change in action["effects"].items():
            self.state[attr] = max(0, min(100, self.state[attr] + change))

        print(f"Состояние после: {self.state}")

# Пример использования
agent = UtilityBasedAgent()
best_action = agent.choose_best_action()
agent.execute_action(best_action)

# =============================================================================
# РАЗДЕЛ 3: ОБУЧАЮЩИЕСЯ АГЕНТЫ
# =============================================================================

# 3.1 Q-Learning агент (простой RL)
import random
import numpy as np

class QLearningAgent:
    """
    Агент с подкрепляющим обучением (Q-Learning)
    Принцип: обучение через взаимодействие и получение наград
    """

    def __init__(self, states, actions, learning_rate=0.1, discount=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

        # Q-таблица: состояние -> действие -> Q-значение
        self.q_table = {}
        for state in states:
            self.q_table[state] = {}
            for action in actions:
                self.q_table[state][action] = 0.0

    def choose_action(self, state):
        """Выбор действия (epsilon-greedy)"""
        if random.random() < self.epsilon:
            # Исследование: случайное действие
            return random.choice(self.actions)
        else:
            # Эксплуатация: лучшее известное действие
            return max(self.actions, key=lambda a: self.q_table[state][a])

    def update_q_value(self, state, action, reward, next_state):
        """Обновление Q-значения"""
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def train(self, environment, episodes=1000):
        """Обучение агента"""
        print("Начинаю обучение...")

        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0

            while not environment.is_done():
                action = self.choose_action(state)
                next_state, reward = environment.step(action)

                self.update_q_value(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            if episode % 100 == 0:
                print(f"Эпизод {episode}: награда = {total_reward}")

        print("Обучение завершено!")
        self.epsilon = 0  # Отключаем исследование после обучения

# Простая среда для демонстрации
class SimpleGridEnvironment:
    """Простая сетка 3x3 с целью в правом нижнем углу"""

    def __init__(self):
        self.size = 3
        self.goal = (2, 2)
        self.current_pos = None
        self.done = False

    def reset(self):
        self.current_pos = (0, 0)
        self.done = False
        return self._get_state()

    def _get_state(self):
        return f"{self.current_pos[0]}_{self.current_pos[1]}"

    def step(self, action):
        if self.done:
            return self._get_state(), 0

        x, y = self.current_pos

        # Движения: up, down, left, right
        if action == "up" and x > 0:
            x -= 1
        elif action == "down" and x < self.size - 1:
            x += 1
        elif action == "left" and y > 0:
            y -= 1
        elif action == "right" and y < self.size - 1:
            y += 1

        self.current_pos = (x, y)

        # Награда
        if self.current_pos == self.goal:
            reward = 100
            self.done = True
        else:
            reward = -1  # Небольшой штраф за каждый шаг

        return self._get_state(), reward

    def is_done(self):
        return self.done

# Пример обучения Q-Learning агента
states = [f"{i}_{j}" for i in range(3) for j in range(3)]
actions = ["up", "down", "left", "right"]

agent = QLearningAgent(states, actions)
environment = SimpleGridEnvironment()

# Обучение
agent.train(environment, episodes=500)

# Тестирование обученного агента
print("\nТестирование обученного агента:")
state = environment.reset()
path = [state]

while not environment.is_done():
    action = agent.choose_action(state)
    print(f"Состояние: {state}, Действие: {action}")

    state, reward = environment.step(action)
    path.append(state)

print(f"Путь: {' -> '.join(path)}")

# =============================================================================
# РАЗДЕЛ 4: LLM-POWERED АГЕНТЫ
# =============================================================================

# 4.1 Простой ReAct агент
class SimpleReActAgent:
    """
    Базовый ReAct агент: Reasoning + Acting
    Принцип: Thought -> Action -> Observation -> Thought...
    """

    def __init__(self, llm_function):
        self.llm = llm_function  # Функция вызова LLM
        self.tools = {
            "calculator": self._calculator,
            "search": self._search,
            "memory": self._memory
        }
        self.memory = []
        self.max_iterations = 5

    def _calculator(self, expression: str) -> str:
        """Простой калькулятор"""
        try:
            result = eval(expression)  # В продакшене использовать безопасный eval
            return f"Результат: {result}"
        except:
            return "Ошибка в вычислениях"

    def _search(self, query: str) -> str:
        """Имитация поиска"""
        # В реальности здесь был бы вызов поисковой системы
        fake_results = {
            "weather": "Сегодня солнечно, 22°C",
            "python": "Python - это язык программирования высокого уровня",
            "AI": "Искусственный интеллект - область информатики"
        }

        for key in fake_results:
            if key.lower() in query.lower():
                return fake_results[key]

        return f"Информация по запросу '{query}' не найдена"

    def _memory(self, action: str) -> str:
        """Работа с памятью"""
        if action.startswith("save:"):
            info = action[5:].strip()
            self.memory.append(info)
            return f"Сохранено: {info}"
        elif action == "recall":
            return f"В памяти: {', '.join(self.memory) if self.memory else 'пусто'}"
        else:
            return "Доступные действия: save:<текст>, recall"

    def parse_action(self, response: str) -> tuple:
        """Парсинг действия из ответа LLM"""
        lines = response.strip().split('\n')

        for line in lines:
            if line.startswith("Action:"):
                action_part = line[7:].strip()
                if "[" in action_part and "]" in action_part:
                    tool = action_part.split("[")[0].strip()
                    args = action_part.split("[")[1].split("]")[0].strip()
                    return tool, args

        return None, None

    def run(self, query: str) -> str:
        """Основной цикл ReAct"""
        prompt = f"""Ты полезный помощник с доступом к инструментам.

Доступные инструменты:
- calculator[выражение] - вычисления
- search[запрос] - поиск информации  
- memory[save:текст или recall] - работа с памятью

Отвечай в формате:
Thought: твои размышления
Action: инструмент[аргументы]
(после получения результата продолжи рассуждения)

Запрос пользователя: {query}"""

        conversation = [prompt]

        for iteration in range(self.max_iterations):
            # Получаем ответ от LLM
            llm_response = self.llm("\n".join(conversation))
            conversation.append(f"Assistant: {llm_response}")

            # Парсим действие
            tool, args = self.parse_action(llm_response)

            if tool and tool in self.tools:
                # Выполняем действие
                result = self.tools[tool](args)
                observation = f"Observation: {result}"
                conversation.append(observation)
                print(f"Действие: {tool}[{args}]")
                print(f"Результат: {result}")
            else:
                # Нет действия - возвращаем финальный ответ
                return llm_response

        return "Превышено максимальное количество итераций"

# Имитация LLM для демонстрации
def mock_llm(prompt: str) -> str:
    """Простая имитация LLM для демонстрации"""
    if "2+2" in prompt or "вычисли" in prompt.lower():
        return """Thought: Мне нужно вычислить математическое выражение.
Action: calculator[2+2]"""
    elif "weather" in prompt.lower() or "погода" in prompt.lower():
        return """Thought: Пользователь спрашивает о погоде, поищу информацию.
Action: search[weather]"""
    else:
        return "Thought: Это простой вопрос, я могу ответить напрямую.\nЯ готов помочь с вашим запросом!"

# Пример использования ReAct агента
agent = SimpleReActAgent(mock_llm)
result = agent.run("Сколько будет 2+2?")
print(f"\nФинальный ответ: {result}")

# =============================================================================
# РАЗДЕЛ 5: МУЛЬТИ-АГЕНТНЫЕ СИСТЕМЫ  
# =============================================================================

# 5.1 Простая мульти-агентная система с супервизором
class Agent:
    """Базовый класс агента"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.skills = []

    def can_handle(self, task_type: str) -> bool:
        """Может ли агент выполнить задачу"""
        return task_type in self.skills

    def execute(self, task: str) -> str:
        """Выполнение задачи"""
        return f"{self.name} выполнил: {task}"

class SpecialistAgent(Agent):
    """Агент-специалист"""

    def __init__(self, name: str, specialization: str, skills: list):
        super().__init__(name, "specialist")
        self.specialization = specialization
        self.skills = skills

    def execute(self, task: str) -> str:
        """Специализированное выполнение"""
        return f"{self.name} ({self.specialization}) обработал: {task}"

class SupervisorAgent(Agent):
    """Агент-супервизор"""

    def __init__(self, name: str):
        super().__init__(name, "supervisor")
        self.team = []
        self.task_queue = []

    def add_agent(self, agent: Agent):
        """Добавление агента в команду"""
        self.team.append(agent)

    def delegate_task(self, task: str, task_type: str) -> str:
        """Делегирование задачи подходящему агенту"""
        # Поиск подходящего агента
        suitable_agents = [agent for agent in self.team if agent.can_handle(task_type)]

        if not suitable_agents:
            return f"Нет агента для задачи типа: {task_type}"

        # Выбираем первого подходящего (можно добавить логику выбора)
        chosen_agent = suitable_agents[0]
        result = chosen_agent.execute(task)

        print(f"Супервизор {self.name} делегировал задачу агенту {chosen_agent.name}")
        return result

    def coordinate_workflow(self, tasks: list) -> list:
        """Координация выполнения множественных задач"""
        results = []

        for task, task_type in tasks:
            print(f"\nОбработка задачи: {task} (тип: {task_type})")
            result = self.delegate_task(task, task_type)
            results.append(result)

        return results

# Создание мульти-агентной системы
print("\n=== МУЛЬТИ-АГЕНТНАЯ СИСТЕМА ===")

# Создаем специализированных агентов
code_agent = SpecialistAgent("CodeBot", "Программирование", ["coding", "debugging", "review"])
data_agent = SpecialistAgent("DataBot", "Анализ данных", ["analysis", "visualization", "statistics"])
doc_agent = SpecialistAgent("DocBot", "Документация", ["writing", "documentation", "editing"])

# Создаем супервизора и формируем команду
supervisor = SupervisorAgent("ProjectManager")
supervisor.add_agent(code_agent)
supervisor.add_agent(data_agent)
supervisor.add_agent(doc_agent)

# Список задач для выполнения
tasks = [
    ("Написать функцию сортировки", "coding"),
    ("Проанализировать продажи за квартал", "analysis"),
    ("Создать README для проекта", "documentation"),
    ("Исправить баг в коде", "debugging"),
    ("Подготовить отчет с графиками", "visualization")
]

# Выполнение задач
results = supervisor.coordinate_workflow(tasks)

print("\n=== РЕЗУЛЬТАТЫ ===")
for i, result in enumerate(results, 1):
    print(f"{i}. {result}")

print("\n✅ Все примеры созданы и готовы к использованию!")
