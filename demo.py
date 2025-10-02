# 🚀 Демонстрация Qwen3 Code Assistant

from qwen3_code_assistant import Qwen3CodeAssistant
import time

def run_demo():
    print("🎯 ДЕМОНСТРАЦИЯ QWEN3 CODE ASSISTANT")
    print("=" * 50)

    # Инициализация
    assistant = Qwen3CodeAssistant(model_name="qwen3")

    # Тестовые запросы
    demos = [
        ("Анализ кода", "Найди проблемы в этом коде", "def login(user, pwd): return user == 'admin'"),
        ("Генерация кода", "Создай функцию для вычисления факториала", ""),
        ("Создание тестов", "Напиши тесты для функции сложения", "def add(a, b): return a + b"),
        ("Документация", "Создай документацию для класса", "class Calculator: pass"),
        ("Рефакторинг", "Улучши этот код", "def process(data): result = []; [result.append(x*2) for x in data]; return result")
    ]

    for i, (title, query, code) in enumerate(demos, 1):
        print(f"\n{i}. {title}")
        print("-" * 30)
        print(f"Запрос: {query}")

        if code:
            print(f"Код: {code}")

        print("⏳ Обработка...")

        start = time.time()
        result = assistant.ask(query, code)
        elapsed = time.time() - start

        if 'error' not in result:
            print(f"✅ Готово за {elapsed:.1f}с")
            print(f"🎯 Тип: {result.get('task_type', 'unknown')}")
            print(f"🤖 Агент: {result.get('agent_used', 'unknown')}")

            # Показываем результат (первые 200 символов)
            for key in ['analysis', 'generated_code', 'tests', 'documentation', 'refactored_code']:
                if result.get(key):
                    content = result[key][:200] + "..." if len(result[key]) > 200 else result[key]
                    print(f"📄 Результат: {content}")
                    break
        else:
            print(f"❌ Ошибка: {result['error']}")

        if i < len(demos):
            input("\nНажмите Enter для продолжения...")

    print("\n🎉 Демонстрация завершена!")

def run_interactive():
    print("🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    assistant = Qwen3CodeAssistant()
    assistant.run_interactive()

def run_benchmark():
    print("⚡ БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    assistant = Qwen3CodeAssistant()

    queries = [
        "Создай простую функцию",
        "Найди баги в коде: def test(): pass",
        "Напиши тест для функции"
    ]

    total_time = 0
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query[:30]}...")
        start = time.time()
        result = assistant.ask(query)
        elapsed = time.time() - start
        total_time += elapsed

        status = "✅" if 'error' not in result else "❌"
        print(f"   {status} {elapsed:.2f}с")

    print(f"\n📊 Среднее время: {total_time/len(queries):.2f}с")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "demo":
            run_demo()
        elif mode == "interactive":
            run_interactive()
        elif mode == "benchmark":
            run_benchmark()
    else:
        print("Режимы: demo, interactive, benchmark")
        choice = input("Выберите режим (demo): ") or "demo"

        if choice == "demo":
            run_demo()
        elif choice == "interactive":
            run_interactive()
        elif choice == "benchmark":
            run_benchmark()
        else:
            run_demo()
