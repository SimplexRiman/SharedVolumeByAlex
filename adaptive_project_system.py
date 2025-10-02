
# 🔄 Система автоматической адаптации к новым проектам
# Интеллектуальная настройка AI помощника под любой проект

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

@dataclass
class ProjectConfig:
    """Конфигурация проекта"""
    name: str
    language: str
    framework: str
    project_type: str
    dependencies: List[str]
    test_framework: Optional[str] = None
    build_tool: Optional[str] = None
    package_manager: Optional[str] = None
    linting_tools: List[str] = None

    def __post_init__(self):
        if self.linting_tools is None:
            self.linting_tools = []

class ProjectDetector(ABC):
    """Базовый класс для детекторов проектов"""

    @abstractmethod
    def detect(self, project_path: str) -> Optional[ProjectConfig]:
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """Приоритет детектора (выше = важнее)"""
        pass

class PythonProjectDetector(ProjectDetector):
    """Детектор Python проектов"""

    def detect(self, project_path: str) -> Optional[ProjectConfig]:
        path = Path(project_path)

        # Поиск Python файлов
        python_files = list(path.rglob("*.py"))
        if not python_files:
            return None

        # Определение типа проекта
        framework = self._detect_framework(path)
        project_type = self._detect_project_type(path, framework)
        dependencies = self._extract_dependencies(path)
        test_framework = self._detect_test_framework(path)
        build_tool = self._detect_build_tool(path)
        linting_tools = self._detect_linting_tools(path)

        return ProjectConfig(
            name=path.name,
            language="Python",
            framework=framework,
            project_type=project_type,
            dependencies=dependencies,
            test_framework=test_framework,
            build_tool=build_tool,
            package_manager="pip" if (path / "requirements.txt").exists() else "poetry" if (path / "pyproject.toml").exists() else "pip",
            linting_tools=linting_tools
        )

    def _detect_framework(self, path: Path) -> str:
        """Определение Python фреймворка"""
        # Django
        if (path / "manage.py").exists() or any(f.name == "settings.py" for f in path.rglob("settings.py")):
            return "Django"

        # Flask
        requirements_content = ""
        if (path / "requirements.txt").exists():
            requirements_content = (path / "requirements.txt").read_text()

        if "flask" in requirements_content.lower():
            return "Flask"

        # FastAPI
        if "fastapi" in requirements_content.lower():
            return "FastAPI"

        # Data Science
        if any(x in requirements_content.lower() for x in ["pandas", "numpy", "matplotlib", "jupyter"]):
            return "Data Science"

        # ML/AI
        if any(x in requirements_content.lower() for x in ["tensorflow", "pytorch", "scikit-learn"]):
            return "Machine Learning"

        return "Pure Python"

    def _detect_project_type(self, path: Path, framework: str) -> str:
        """Определение типа проекта"""
        if framework in ["Django", "Flask", "FastAPI"]:
            return "Web Application"
        elif framework == "Data Science":
            return "Data Analysis"
        elif framework == "Machine Learning":
            return "ML/AI Project"
        elif any(f.name == "__main__.py" for f in path.rglob("__main__.py")):
            return "CLI Application"
        elif any(f.name == "setup.py" for f in path.rglob("setup.py")):
            return "Python Package"
        else:
            return "General Purpose"

    def _extract_dependencies(self, path: Path) -> List[str]:
        """Извлечение зависимостей"""
        dependencies = []

        if (path / "requirements.txt").exists():
            content = (path / "requirements.txt").read_text()
            for line in content.split("\n"):
                if line.strip() and not line.startswith("#"):
                    dep = line.split("==")[0].split(">=")[0].split("~=")[0].strip()
                    dependencies.append(dep)

        return dependencies[:20]  # Ограничиваем количество

    def _detect_test_framework(self, path: Path) -> Optional[str]:
        """Определение тестового фреймворка"""
        if any(f.name.startswith("test_") for f in path.rglob("test_*.py")):
            return "pytest"
        elif any(f.name.endswith("_test.py") for f in path.rglob("*_test.py")):
            return "unittest"
        elif any("tests" in str(f) for f in path.rglob("*.py")):
            return "pytest"  # по умолчанию
        return None

    def _detect_build_tool(self, path: Path) -> Optional[str]:
        """Определение инструмента сборки"""
        if (path / "pyproject.toml").exists():
            return "poetry"
        elif (path / "setup.py").exists():
            return "setuptools"
        elif (path / "Pipfile").exists():
            return "pipenv"
        return None

    def _detect_linting_tools(self, path: Path) -> List[str]:
        """Определение инструментов линтинга"""
        tools = []

        # Проверка конфигурационных файлов
        if (path / ".pylintrc").exists() or (path / "pylint.cfg").exists():
            tools.append("pylint")
        if (path / ".flake8").exists() or (path / "setup.cfg").exists():
            tools.append("flake8")
        if (path / "pyproject.toml").exists():
            content = (path / "pyproject.toml").read_text()
            if "black" in content:
                tools.append("black")
            if "mypy" in content:
                tools.append("mypy")

        return tools

    def get_priority(self) -> int:
        return 100

class JavaScriptProjectDetector(ProjectDetector):
    """Детектор JavaScript/TypeScript проектов"""

    def detect(self, project_path: str) -> Optional[ProjectConfig]:
        path = Path(project_path)

        # Поиск package.json
        package_json_path = path / "package.json"
        if not package_json_path.exists():
            return None

        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
        except:
            return None

        # Определение языка
        language = "TypeScript" if (path / "tsconfig.json").exists() else "JavaScript"

        # Определение фреймворка
        framework = self._detect_js_framework(package_data, path)
        project_type = self._detect_js_project_type(package_data, framework)
        test_framework = self._detect_js_test_framework(package_data)
        build_tool = self._detect_js_build_tool(package_data, path)

        dependencies = list(package_data.get("dependencies", {}).keys())[:15]

        return ProjectConfig(
            name=package_data.get("name", path.name),
            language=language,
            framework=framework,
            project_type=project_type,
            dependencies=dependencies,
            test_framework=test_framework,
            build_tool=build_tool,
            package_manager="npm" if (path / "package-lock.json").exists() else "yarn" if (path / "yarn.lock").exists() else "npm",
            linting_tools=self._detect_js_linting_tools(package_data, path)
        )

    def _detect_js_framework(self, package_data: Dict, path: Path) -> str:
        """Определение JavaScript фреймворка"""
        deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}

        if "react" in deps:
            if "next" in deps:
                return "Next.js"
            elif "gatsby" in deps:
                return "Gatsby"
            else:
                return "React"
        elif "vue" in deps:
            if "nuxt" in deps:
                return "Nuxt.js"
            else:
                return "Vue.js"
        elif "angular" in deps or "@angular/core" in deps:
            return "Angular"
        elif "express" in deps:
            return "Express.js"
        elif "svelte" in deps:
            return "Svelte"
        elif "solid-js" in deps:
            return "SolidJS"

        return "Vanilla JavaScript"

    def _detect_js_project_type(self, package_data: Dict, framework: str) -> str:
        """Определение типа JavaScript проекта"""
        if framework in ["React", "Vue.js", "Angular", "Next.js", "Nuxt.js", "Gatsby", "Svelte", "SolidJS"]:
            return "Frontend Application"
        elif framework == "Express.js":
            return "Backend API"
        elif "electron" in package_data.get("dependencies", {}):
            return "Desktop Application"
        elif package_data.get("type") == "module":
            return "ES Module"
        else:
            return "Node.js Application"

    def _detect_js_test_framework(self, package_data: Dict) -> Optional[str]:
        """Определение тестового фреймворка"""
        deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}

        if "jest" in deps:
            return "Jest"
        elif "mocha" in deps:
            return "Mocha"
        elif "vitest" in deps:
            return "Vitest"
        elif "cypress" in deps:
            return "Cypress"
        elif "playwright" in deps:
            return "Playwright"

        return None

    def _detect_js_build_tool(self, package_data: Dict, path: Path) -> Optional[str]:
        """Определение инструмента сборки"""
        deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}

        if (path / "webpack.config.js").exists() or "webpack" in deps:
            return "Webpack"
        elif (path / "vite.config.js").exists() or (path / "vite.config.ts").exists():
            return "Vite"
        elif (path / "rollup.config.js").exists():
            return "Rollup"
        elif "parcel" in deps:
            return "Parcel"

        return None

    def _detect_js_linting_tools(self, package_data: Dict, path: Path) -> List[str]:
        """Определение линтинг инструментов"""
        tools = []
        deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}

        if "eslint" in deps or (path / ".eslintrc.js").exists():
            tools.append("ESLint")
        if "prettier" in deps or (path / ".prettierrc").exists():
            tools.append("Prettier")
        if "typescript" in deps:
            tools.append("TypeScript")

        return tools

    def get_priority(self) -> int:
        return 95

class JavaProjectDetector(ProjectDetector):
    """Детектор Java проектов"""

    def detect(self, project_path: str) -> Optional[ProjectConfig]:
        path = Path(project_path)

        # Поиск Java файлов
        java_files = list(path.rglob("*.java"))
        if not java_files:
            return None

        framework = self._detect_java_framework(path)
        project_type = self._detect_java_project_type(path, framework)
        build_tool = self._detect_java_build_tool(path)
        test_framework = self._detect_java_test_framework(path)

        return ProjectConfig(
            name=path.name,
            language="Java",
            framework=framework,
            project_type=project_type,
            dependencies=self._extract_java_dependencies(path, build_tool),
            test_framework=test_framework,
            build_tool=build_tool,
            package_manager=build_tool,
            linting_tools=self._detect_java_linting_tools(path)
        )

    def _detect_java_framework(self, path: Path) -> str:
        """Определение Java фреймворка"""
        # Spring Boot
        if any("spring-boot" in str(f) for f in path.rglob("*.xml")) or any("@SpringBootApplication" in f.read_text(errors='ignore') for f in path.rglob("*.java")):
            return "Spring Boot"

        # Spring
        if any("spring" in str(f).lower() for f in path.rglob("*.xml")):
            return "Spring"

        # Android
        if (path / "AndroidManifest.xml").exists():
            return "Android"

        return "Core Java"

    def _detect_java_project_type(self, path: Path, framework: str) -> str:
        """Определение типа Java проекта"""
        if framework == "Spring Boot":
            return "Web Application"
        elif framework == "Android":
            return "Mobile Application"
        elif any("junit" in str(f).lower() for f in path.rglob("*.xml")):
            return "Library/Framework"
        else:
            return "Console Application"

    def _detect_java_build_tool(self, path: Path) -> Optional[str]:
        """Определение инструмента сборки"""
        if (path / "pom.xml").exists():
            return "Maven"
        elif (path / "build.gradle").exists() or (path / "build.gradle.kts").exists():
            return "Gradle"
        elif (path / "build.xml").exists():
            return "Ant"
        return None

    def _detect_java_test_framework(self, path: Path) -> Optional[str]:
        """Определение тестового фреймворка"""
        # Поиск в build файлах или Java классах
        for file in path.rglob("*.xml"):
            try:
                content = file.read_text(errors='ignore').lower()
                if "junit" in content:
                    return "JUnit"
                elif "testng" in content:
                    return "TestNG"
            except:
                continue

        for file in path.rglob("*.java"):
            try:
                content = file.read_text(errors='ignore')
                if "@Test" in content:
                    return "JUnit" if "import org.junit" in content else "TestNG"
            except:
                continue

        return None

    def _extract_java_dependencies(self, path: Path, build_tool: Optional[str]) -> List[str]:
        """Извлечение Java зависимостей"""
        dependencies = []

        if build_tool == "Maven" and (path / "pom.xml").exists():
            # Упрощенное извлечение из Maven POM
            try:
                content = (path / "pom.xml").read_text()
                # Ищем основные фреймворки в pom.xml
                if "spring-boot" in content:
                    dependencies.append("Spring Boot")
                if "junit" in content:
                    dependencies.append("JUnit")
                if "hibernate" in content:
                    dependencies.append("Hibernate")
            except:
                pass

        return dependencies

    def _detect_java_linting_tools(self, path: Path) -> List[str]:
        """Определение Java линтинг инструментов"""
        tools = []

        if (path / "checkstyle.xml").exists():
            tools.append("Checkstyle")
        if (path / "spotbugs.xml").exists():
            tools.append("SpotBugs")
        if (path / "pmd.xml").exists():
            tools.append("PMD")

        return tools

    def get_priority(self) -> int:
        return 90

# Аналогично можно создать детекторы для других языков...

class ProjectAnalyzer:
    """Анализатор проектов с автоматическим определением типа"""

    def __init__(self):
        self.detectors: List[ProjectDetector] = [
            PythonProjectDetector(),
            JavaScriptProjectDetector(),
            JavaProjectDetector(),
            # Можно добавить больше детекторов
        ]
        # Сортировка по приоритету
        self.detectors.sort(key=lambda x: x.get_priority(), reverse=True)

    def analyze_project(self, project_path: str) -> Optional[ProjectConfig]:
        """Анализ проекта и автоматическое определение конфигурации"""
        for detector in self.detectors:
            try:
                config = detector.detect(project_path)
                if config:
                    return config
            except Exception as e:
                print(f"Ошибка в детекторе {detector.__class__.__name__}: {e}")

        return None

    def generate_ai_config(self, project_config: ProjectConfig) -> Dict[str, Any]:
        """Генерация конфигурации AI системы на основе проекта"""

        base_config = {
            "project": {
                "name": project_config.name,
                "language": project_config.language,
                "framework": project_config.framework,
                "type": project_config.project_type
            },
            "agents": self._configure_agents(project_config),
            "llm": self._configure_llm(project_config),
            "vector_db": self._configure_vector_db(project_config),
            "analysis": self._configure_analysis(project_config)
        }

        return base_config

    def _configure_agents(self, config: ProjectConfig) -> Dict[str, Any]:
        """Настройка агентов под тип проекта"""
        agents = {
            "supervisor": {
                "model": "qwen2.5-coder:7b",
                "prompt_template": f"Вы ведущий AI архитектор для {config.framework} проектов..."
            },
            "code_analyzer": {
                "specialization": f"{config.language} {config.framework} analysis",
                "tools": self._get_analysis_tools(config)
            },
            "code_generator": {
                "specialization": f"{config.language} {config.framework} development",
                "style_guide": self._get_style_guide(config)
            }
        }

        # Добавление специализированных агентов
        if config.project_type == "Web Application":
            agents["api_expert"] = {
                "specialization": "REST API design and testing",
                "tools": ["swagger_analyzer", "api_tester"]
            }
        elif config.project_type == "Data Analysis":
            agents["data_expert"] = {
                "specialization": "Data analysis and visualization",
                "tools": ["pandas_analyzer", "plot_generator"]
            }

        return agents

    def _get_analysis_tools(self, config: ProjectConfig) -> List[str]:
        """Получение инструментов анализа для языка"""
        tools = ["ast_parser", "dependency_analyzer"]

        if config.language == "Python":
            tools.extend(["pylint_integration", "mypy_checker"])
        elif config.language == "JavaScript":
            tools.extend(["eslint_integration", "typescript_checker"])
        elif config.language == "Java":
            tools.extend(["checkstyle_integration", "spotbugs_analyzer"])

        return tools

    def _get_style_guide(self, config: ProjectConfig) -> Dict[str, str]:
        """Получение руководства по стилю"""
        style_guides = {
            "Python": {
                "Django": "Django coding style with PEP 8",
                "Flask": "Flask best practices with PEP 8",
                "Data Science": "Data science conventions with type hints"
            },
            "JavaScript": {
                "React": "React + ESLint + Prettier standards",
                "Vue.js": "Vue.js style guide + ESLint",
                "Node.js": "Node.js best practices + ESLint"
            },
            "Java": {
                "Spring Boot": "Spring Boot conventions + Google Java Style",
                "Android": "Android development guidelines"
            }
        }

        return {
            "language": config.language,
            "framework_specific": style_guides.get(config.language, {}).get(config.framework, "Standard conventions")
        }

    def _configure_llm(self, config: ProjectConfig) -> Dict[str, Any]:
        """Настройка LLM под проект"""
        llm_config = {
            "provider": "ollama",
            "models": {
                "primary": "qwen2.5-coder:7b",
                "specialized": []
            }
        }

        # Специализированные модели для разных типов проектов
        if config.project_type == "Web Application":
            llm_config["models"]["specialized"].append("codellama:7b")  # Для API генерации
        elif config.project_type == "Data Analysis":
            llm_config["models"]["specialized"].append("deepseek-coder")  # Для data science

        return llm_config

    def _configure_vector_db(self, config: ProjectConfig) -> Dict[str, Any]:
        """Настройка векторной БД под размер проекта"""
        # Оценка размера проекта по количеству зависимостей
        estimated_size = len(config.dependencies) * 50  # Примерная оценка

        if estimated_size < 1000:
            return {
                "type": "chromadb",
                "config": {"persist_directory": f"./embeddings/{config.name}"}
            }
        else:
            return {
                "type": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": f"{config.name}_code"
                }
            }

    def _configure_analysis(self, config: ProjectConfig) -> Dict[str, Any]:
        """Настройка анализа кода под язык"""
        analysis_config = {
            "chunking": {
                "max_size": 500,
                "overlap": 50,
                "respect_boundaries": True
            },
            "languages": [config.language.lower()],
            "exclude_patterns": [
                ".git/**", "node_modules/**", "__pycache__/**",
                "*.pyc", "*.class", "target/**", "build/**"
            ]
        }

        # Языко-специфичные настройки
        if config.language == "Python":
            analysis_config["include_patterns"] = ["*.py"]
            analysis_config["extract_docstrings"] = True
        elif config.language == "JavaScript":
            analysis_config["include_patterns"] = ["*.js", "*.ts", "*.jsx", "*.tsx"]
            analysis_config["extract_jsdoc"] = True
        elif config.language == "Java":
            analysis_config["include_patterns"] = ["*.java"]
            analysis_config["extract_javadoc"] = True

        return analysis_config

class AdaptiveSystemManager:
    """Менеджер адаптивной системы"""

    def __init__(self):
        self.analyzer = ProjectAnalyzer()
        self.current_config: Optional[ProjectConfig] = None
        self.config_cache = {}

    def adapt_to_project(self, project_path: str) -> Dict[str, Any]:
        """Автоматическая адаптация к новому проекту"""
        print(f"🔍 Анализ проекта: {project_path}")

        # Проверка кэша
        project_hash = self._get_project_hash(project_path)
        if project_hash in self.config_cache:
            print("📋 Использование кэшированной конфигурации")
            return self.config_cache[project_hash]

        # Анализ проекта
        project_config = self.analyzer.analyze_project(project_path)

        if not project_config:
            print("❌ Не удалось определить тип проекта")
            return self._get_default_config()

        print(f"✅ Обнаружен проект: {project_config.language} {project_config.framework}")
        print(f"📊 Тип: {project_config.project_type}")
        print(f"🔧 Зависимости: {len(project_config.dependencies)} библиотек")

        # Генерация AI конфигурации
        ai_config = self.analyzer.generate_ai_config(project_config)

        # Сохранение в кэш
        self.config_cache[project_hash] = ai_config
        self.current_config = project_config

        # Сохранение конфигурации
        self._save_project_config(project_path, ai_config)

        return ai_config

    def _get_project_hash(self, project_path: str) -> str:
        """Генерация хэша проекта для кэширования"""
        import hashlib

        # Используем mtime ключевых файлов для определения изменений
        key_files = [
            "package.json", "requirements.txt", "pom.xml", 
            "Cargo.toml", "go.mod", "composer.json"
        ]

        file_info = []
        for file_name in key_files:
            file_path = Path(project_path) / file_name
            if file_path.exists():
                file_info.append(f"{file_name}:{file_path.stat().st_mtime}")

        hash_string = f"{project_path}:{'|'.join(file_info)}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _save_project_config(self, project_path: str, config: Dict[str, Any]):
        """Сохранение конфигурации проекта"""
        config_path = Path(project_path) / ".ai_assistant" / "config.yaml"
        config_path.parent.mkdir(exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"💾 Конфигурация сохранена: {config_path}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            "project": {
                "name": "Unknown",
                "language": "Mixed",
                "framework": "Unknown",
                "type": "General"
            },
            "agents": {
                "supervisor": {"model": "qwen2.5-coder:7b"},
                "code_analyzer": {"specialization": "General code analysis"},
                "code_generator": {"specialization": "Multi-language code generation"}
            },
            "llm": {"provider": "ollama", "model": "qwen2.5-coder:7b"},
            "vector_db": {"type": "chromadb"},
            "analysis": {"languages": ["python", "javascript", "java"]}
        }

    def get_project_specific_prompts(self) -> Dict[str, str]:
        """Получение специфичных для проекта промптов"""
        if not self.current_config:
            return {}

        prompts = {
            "supervisor": f"""
            Вы - ведущий AI архитектор для {self.current_config.framework} проектов.
            Специализация: {self.current_config.project_type} на {self.current_config.language}.

            Команда экспертов:
            - code_analyzer: анализ {self.current_config.language} кода
            - code_generator: генерация {self.current_config.framework} кода
            - test_specialist: создание {self.current_config.test_framework or 'подходящих'} тестов

            Учитывайте особенности {self.current_config.framework} при делегировании задач.
            """,

            "code_analyzer": f"""
            Вы - эксперт по анализу {self.current_config.language} кода и {self.current_config.framework} архитектуре.

            Специализация:
            - Анализ паттернов {self.current_config.framework}
            - Проверка best practices для {self.current_config.project_type}
            - Обнаружение anti-patterns и технического долга

            Используемые инструменты: {', '.join(self.current_config.linting_tools)}
            """,

            "code_generator": f"""
            Вы - эксперт по разработке {self.current_config.language} приложений на {self.current_config.framework}.

            Следуйте конвенциям:
            - {self.current_config.language} coding standards
            - {self.current_config.framework} best practices
            - Паттерны проектирования для {self.current_config.project_type}

            Генерируйте код, совместимый с: {', '.join(self.current_config.dependencies[:5])}
            """
        }

        return prompts

# ИНТЕГРАЦИЯ С VISUAL STUDIO

class VisualStudioIntegration:
    """Интеграция с Visual Studio через Language Server Protocol"""

    def __init__(self, adaptive_manager: AdaptiveSystemManager):
        self.adaptive_manager = adaptive_manager
        self.lsp_config = {}

    def create_vs_extension_config(self, project_path: str) -> Dict[str, Any]:
        """Создание конфигурации для Visual Studio Extension"""

        # Адаптация к проекту
        ai_config = self.adaptive_manager.adapt_to_project(project_path)

        # Конфигурация LSP сервера
        lsp_config = {
            "name": "AI Code Assistant",
            "displayName": "AI Code Assistant for " + ai_config["project"]["framework"],
            "description": f"Intelligent assistant for {ai_config['project']['language']} {ai_config['project']['framework']} projects",
            "version": "1.0.0",
            "engines": {
                "vscode": "^1.74.0"
            },
            "categories": ["Other", "Machine Learning"],
            "activationEvents": [
                f"onLanguage:{ai_config['project']['language'].lower()}",
                "onCommand:ai-assistant.analyze",
                "onCommand:ai-assistant.generate"
            ],
            "main": "./out/extension.js",
            "contributes": {
                "commands": [
                    {
                        "command": "ai-assistant.analyze",
                        "title": "Analyze Code",
                        "category": "AI Assistant"
                    },
                    {
                        "command": "ai-assistant.generate", 
                        "title": "Generate Code",
                        "category": "AI Assistant"
                    },
                    {
                        "command": "ai-assistant.explain",
                        "title": "Explain Code",
                        "category": "AI Assistant"
                    }
                ],
                "configuration": {
                    "title": "AI Code Assistant",
                    "properties": {
                        "aiAssistant.model": {
                            "type": "string",
                            "default": ai_config["llm"]["model"],
                            "description": "LLM model to use"
                        },
                        "aiAssistant.projectType": {
                            "type": "string",
                            "default": ai_config["project"]["type"],
                            "description": "Project type for optimization"
                        }
                    }
                }
            }
        }

        return lsp_config

    def generate_lsp_ai_config(self, project_config: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация конфигурации для LSP-AI"""

        return {
            "initializationOptions": {
                "memory": {
                    "file_store": {}
                },
                "models": {
                    "completion_model": {
                        "type": "ollama",
                        "model": project_config["llm"]["model"],
                        "url": "http://localhost:11434",
                        "auth_token": None
                    },
                    "chat_model": {
                        "type": "ollama", 
                        "model": project_config["llm"]["model"],
                        "url": "http://localhost:11434",
                        "auth_token": None
                    }
                },
                "completion": {
                    "model": "completion_model",
                    "parameters": {
                        "max_context": 1024,
                        "max_tokens": 64
                    }
                },
                "chat": {
                    "model": "chat_model",
                    "parameters": {
                        "max_context": 4096,
                        "max_tokens": 512
                    }
                }
            }
        }

# АВТОМАТИЧЕСКОЕ ОБНАРУЖЕНИЕ И НАСТРОЙКА

def auto_setup_project(project_path: str) -> str:
    """Автоматическая настройка AI помощника для проекта"""

    print("🚀 Автоматическая настройка AI помощника...")

    # Создание адаптивного менеджера
    adaptive_manager = AdaptiveSystemManager()

    # Анализ и адаптация
    config = adaptive_manager.adapt_to_project(project_path)

    # Создание Visual Studio интеграции
    vs_integration = VisualStudioIntegration(adaptive_manager)
    vs_config = vs_integration.create_vs_extension_config(project_path)
    lsp_config = vs_integration.generate_lsp_ai_config(config)

    # Сохранение конфигураций
    config_dir = Path(project_path) / ".ai_assistant"
    config_dir.mkdir(exist_ok=True)

    # Основная конфигурация
    with open(config_dir / "ai_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # LSP конфигурация
    with open(config_dir / "lsp_config.json", 'w', encoding='utf-8') as f:
        json.dump(lsp_config, f, indent=2, ensure_ascii=False)

    # VS Code конфигурация
    with open(config_dir / "vscode_extension.json", 'w', encoding='utf-8') as f:
        json.dump(vs_config, f, indent=2, ensure_ascii=False)

    # Создание скрипта быстрого запуска
    launch_script = f"""#!/bin/bash
# Скрипт автоматического запуска AI помощника

echo "🚀 Запуск AI помощника для {config['project']['name']}"

# Проверка Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama не установлен. Установите: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Запуск модели
echo "🤖 Загрузка модели {config['llm']['model']}..."
ollama pull {config['llm']['model']}

# Запуск LSP-AI сервера
echo "🔌 Запуск LSP-AI сервера..."
lsp-ai --config .ai_assistant/lsp_config.json

echo "✅ AI помощник готов к работе!"
"""

    with open(config_dir / "launch.sh", 'w', encoding='utf-8') as f:
        f.write(launch_script)

    # Делаем скрипт исполняемым
    os.chmod(config_dir / "launch.sh", 0o755)

    summary = f"""
✅ Автоматическая настройка завершена!

Обнаружен проект:
  📝 Язык: {config['project']['language']}
  🛠️ Фреймворк: {config['project']['framework']}  
  📊 Тип: {config['project']['type']}
  📦 Зависимости: {len(config.get('project', {}).get('dependencies', []))} шт.

Создано:
  📄 ai_config.yaml - основная конфигурация
  🔌 lsp_config.json - настройки LSP-AI
  🎨 vscode_extension.json - конфигурация VS Code
  🚀 launch.sh - скрипт запуска

Для запуска: cd {project_path}/.ai_assistant && ./launch.sh
"""

    return summary

# ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    # Автоматическая настройка для текущего проекта
    result = auto_setup_project("./")
    print(result)
