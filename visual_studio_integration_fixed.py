# 🎨 Интеграция AI помощника с Visual Studio
# Полная интеграция через Language Server Protocol и VS Extensions

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import sys

class VisualStudioExtensionGenerator:
    """Генератор расширений для Visual Studio"""

    def __init__(self):
        self.template_path = Path("./vs_extension_template")
        self.output_path = Path("./generated_extension")

    def generate_extension(self, project_config: Dict[str, Any], extension_name: str = "AICodeAssistant") -> str:
        """Генерация расширения Visual Studio"""

        print(f"🎨 Генерация расширения Visual Studio: {extension_name}")

        # Создание структуры проекта
        self._create_project_structure(extension_name)

        # Генерация манифеста расширения
        manifest_content = self._generate_extension_manifest(project_config, extension_name)

        # Генерация основного кода расширения
        extension_code = self._generate_extension_code(project_config)

        # Генерация конфигурации LSP
        lsp_config = self._generate_lsp_configuration(project_config)

        # Генерация файлов проекта
        project_file = self._generate_project_file(extension_name)

        # Сохранение всех файлов
        self._save_extension_files(extension_name, {
            "source.extension.vsixmanifest": manifest_content,
            "AICodeAssistantPackage.cs": extension_code,
            "lsp-config.json": json.dumps(lsp_config, indent=2),
            f"{extension_name}.csproj": project_file
        })

        # Создание скрипта сборки
        build_script = self._create_build_script(extension_name)

        return str(self.output_path / extension_name)

    def _create_project_structure(self, extension_name: str):
        """Создание структуры проекта расширения"""
        base_path = self.output_path / extension_name

        # Основные директории
        directories = [
            base_path,
            base_path / "Properties",
            base_path / "Resources", 
            base_path / "Commands",
            base_path / "LanguageServer",
            base_path / "UI"
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"📁 Структура проекта создана: {base_path}")

    def _generate_extension_manifest(self, project_config: Dict[str, Any], extension_name: str) -> str:
        """Генерация манифеста VSIX"""

        project_info = project_config.get("project", {})

        # Формируем манифест по частям чтобы избежать проблем с кавычками
        manifest_parts = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">',
            '  <Metadata>',
            f'    <Identity Id="{extension_name}.{project_info.get("language", "Universal")}" Version="1.0" Language="en-US" Publisher="AICodeAssistant" />',
            f'    <DisplayName>AI Code Assistant for {project_info.get("framework", "Development")}</DisplayName>',
            f'    <Description>Intelligent AI assistant for {project_info.get("language", "multi-language")} {project_info.get("framework", "")} development</Description>',
            '    <MoreInfo>https://github.com/ai-code-assistant</MoreInfo>',
            '    <License>LICENSE</License>',
            '    <GettingStartedGuide>README.md</GettingStartedGuide>',
            '    <Icon>Resources\Icon.png</Icon>',
            '    <PreviewImage>Resources\Preview.png</PreviewImage>',
            f'    <Tags>AI, Code Generation, {project_info.get("language", "")}, {project_info.get("framework", "")}</Tags>',
            '  </Metadata>',
            '  <Installation>',
            '    <InstallationTarget Id="Microsoft.VisualStudio.Community" Version="[17.0,18.0)" />',
            '    <InstallationTarget Id="Microsoft.VisualStudio.Pro" Version="[17.0,18.0)" />',
            '    <InstallationTarget Id="Microsoft.VisualStudio.Enterprise" Version="[17.0,18.0)" />',
            '  </Installation>',
            '  <Dependencies>',
            '    <Dependency Id="Microsoft.Framework.NDP" DisplayName="Microsoft .NET Framework" Version="[4.8,)" />',
            '  </Dependencies>',
            '  <Prerequisites>',
            '    <Prerequisite Id="Microsoft.VisualStudio.Component.CoreEditor" Version="[17.0,18.0)" DisplayName="Visual Studio core editor" />',
            '  </Prerequisites>',
            '  <Assets>',
            f'    <Asset Type="Microsoft.VisualStudio.VsPackage" Path="{extension_name}.pkgdef" />',
            f'    <Asset Type="Microsoft.VisualStudio.MefComponent" Path="{extension_name}.dll" />',
            '  </Assets>',
            '</PackageManifest>'
        ]

        return '\n'.join(manifest_parts)

    def _generate_extension_code(self, project_config: Dict[str, Any]) -> str:
        """Генерация основного кода расширения"""

        project_info = project_config.get("project", {})

        extension_code = f"""using System;
using System.ComponentModel.Design;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.VisualStudio.Shell;
using Microsoft.VisualStudio.Shell.Interop;
using Microsoft.VisualStudio.LanguageServer.Client;
using Microsoft.VisualStudio.Utilities;
using System.ComponentModel.Composition;
using System.IO;
using System.Diagnostics;

namespace AICodeAssistant
{{
    /// <summary>
    /// AI Code Assistant Package for Visual Studio
    /// Specialized for {project_info.get('language', 'Multi-language')} {project_info.get('framework', '')} development
    /// </summary>
    [PackageRegistration(UseManagedResourcesOnly = true, AllowsBackgroundLoading = true)]
    [Guid(AICodeAssistantPackage.PackageGuidString)]
    [ProvideMenuResource("Menus.ctmenu", 1)]
    [ProvideService(typeof(AILanguageServerService), IsAsyncQueryable = true)]
    public sealed class AICodeAssistantPackage : AsyncPackage
    {{
        public const string PackageGuidString = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";

        public AICodeAssistantPackage()
        {{
        }}

        protected override async Task InitializeAsync(CancellationToken cancellationToken, IProgress<ServiceProgressData> progress)
        {{
            await base.InitializeAsync(cancellationToken, progress);

            // Add AI service
            AddService(typeof(AILanguageServerService), CreateAILanguageServerService, true);

            // Initialize commands
            await AnalyzeCodeCommand.InitializeAsync(this);
            await GenerateCodeCommand.InitializeAsync(this);
            await ExplainCodeCommand.InitializeAsync(this);
        }}

        private async Task<object> CreateAILanguageServerService(IAsyncServiceContainer container, CancellationToken cancellationToken, Type serviceType)
        {{
            return new AILanguageServerService();
        }}
    }}

    /// <summary>
    /// AI Language Server Service for {project_info.get('framework', 'development')}
    /// </summary>
    public class AILanguageServerService
    {{
        public AILanguageServerService()
        {{
            // Initialize AI service for {project_info.get('language', 'multi-language')} projects
        }}
    }}

    /// <summary>
    /// Language Server Client for AI Assistant
    /// Specialized for {project_info.get('language', 'Multi-language')} development
    /// </summary>
    [ContentType("{project_info.get('language', 'text').lower()}")]
    [Export(typeof(ILanguageClient))]
    [RunOnContext(RunningContext.RunOnHost)]
    public class AILanguageClient : ILanguageClient, ILanguageClientCustomMessage2
    {{
        public string Name => "AI Code Assistant for {project_info.get('framework', 'Development')}";

        public IEnumerable<string> ConfigurationSections {{ get; }} = new[] {{ "aiAssistant" }};

        public object InitializationOptions => new
        {{
            projectType = "{project_info.get('type', 'General')}",
            language = "{project_info.get('language', 'Multi-language')}",
            framework = "{project_info.get('framework', 'None')}",
            specialization = "{project_info.get('language', 'Multi-language')} {project_info.get('framework', '')} development"
        }};

        public IEnumerable<string> FilesToWatch => GetFilePatterns();

        public event AsyncEventHandler<EventArgs> StartAsync;
        public event AsyncEventHandler<EventArgs> StopAsync;

        private IEnumerable<string> GetFilePatterns()
        {{
            var patterns = new List<string> {{ "**/package.json", "**/requirements.txt", "**/pom.xml" }};

            string language = "{project_info.get('language', 'text').lower()}";
            switch(language)
            {{
                case "python":
                    patterns.Add("**/*.py");
                    break;
                case "javascript":
                    patterns.AddRange(new[] {{ "**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx" }});
                    break;
                case "java":
                    patterns.Add("**/*.java");
                    break;
                case "c#":
                    patterns.AddRange(new[] {{ "**/*.cs", "**/*.csx" }});
                    break;
                default:
                    patterns.Add("**/*.*");
                    break;
            }}

            return patterns;
        }}

        public async Task<Connection> ActivateAsync(CancellationToken token)
        {{
            try
            {{
                var serverPath = GetLSPAIPath();
                var configPath = GetConfigPath();

                var startInfo = new ProcessStartInfo
                {{
                    FileName = serverPath,
                    Arguments = $"--config {{configPath}}",
                    UseShellExecute = false,
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }};

                var process = Process.Start(startInfo);

                return new Connection(process.StandardOutput.BaseStream, process.StandardInput.BaseStream);
            }}
            catch (Exception ex)
            {{
                System.Diagnostics.Debug.WriteLine($"Failed to start AI Language Server: {{ex.Message}}");
                return null;
            }}
        }}

        private string GetLSPAIPath()
        {{
            var paths = new[]
            {{
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cargo", "bin", "lsp-ai.exe"),
                "lsp-ai.exe",
                "lsp-ai"
            }};

            foreach (var path in paths)
            {{
                if (File.Exists(path) || IsInPath(path))
                {{
                    return path;
                }}
            }}

            throw new FileNotFoundException("LSP-AI executable not found. Install with: cargo install lsp-ai");
        }}

        private bool IsInPath(string executable)
        {{
            try
            {{
                var process = Process.Start(new ProcessStartInfo
                {{
                    FileName = "where",
                    Arguments = executable,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                }});

                process?.WaitForExit();
                return process?.ExitCode == 0;
            }}
            catch
            {{
                return false;
            }}
        }}

        private string GetConfigPath()
        {{
            var configDir = Path.Combine(Environment.CurrentDirectory, ".ai_assistant");
            var configPath = Path.Combine(configDir, "lsp_config.json");

            if (!File.Exists(configPath))
            {{
                Directory.CreateDirectory(configDir);
                var defaultConfig = GetDefaultConfig();
                File.WriteAllText(configPath, defaultConfig);
            }}

            return configPath;
        }}

        private string GetDefaultConfig()
        {{
            var config = new
            {{
                memory = new {{ file_store = new {{ }} }},
                models = new
                {{
                    completion_model = new
                    {{
                        type = "ollama",
                        model = "qwen2.5-coder:7b",
                        url = "http://localhost:11434"
                    }}
                }},
                completion = new
                {{
                    model = "completion_model",
                    parameters = new
                    {{
                        max_context = 2048,
                        max_tokens = 128,
                        language_hint = "{project_info.get('language', 'multi-language')}",
                        framework_hint = "{project_info.get('framework', 'general')}"
                    }}
                }}
            }};

            return System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions
            {{
                WriteIndented = true
            }});
        }}

        public async Task OnLoadedAsync()
        {{
            await StartAsync?.InvokeAsync(this, EventArgs.Empty);
        }}

        public Task OnServerInitializeFailedAsync(Exception e)
        {{
            return Task.CompletedTask;
        }}

        public Task OnServerInitializedAsync()
        {{
            return Task.CompletedTask;
        }}

        public Task AttachForCustomMessageAsync(JsonRpc rpc)
        {{
            return Task.CompletedTask;
        }}
    }}

    /// <summary>
    /// Analyze Code Command for {project_info.get('language', 'Multi-language')} projects
    /// </summary>
    internal sealed class AnalyzeCodeCommand
    {{
        public const int CommandId = 0x0100;
        public static readonly Guid CommandSet = new Guid("a1b2c3d4-e5f6-7890-abcd-ef1234567891");

        private readonly AsyncPackage package;

        private AnalyzeCodeCommand(AsyncPackage package, OleMenuCommandService commandService)
        {{
            this.package = package ?? throw new ArgumentNullException(nameof(package));
            commandService = commandService ?? throw new ArgumentNullException(nameof(commandService));

            var menuCommandID = new CommandID(CommandSet, CommandId);
            var menuItem = new MenuCommand(this.Execute, menuCommandID);
            commandService.AddCommand(menuItem);
        }}

        public static AnalyzeCodeCommand Instance {{ get; private set; }}

        private Microsoft.VisualStudio.Shell.IAsyncServiceProvider ServiceProvider => this.package;

        public static async Task InitializeAsync(AsyncPackage package)
        {{
            await ThreadHelper.JoinableTaskFactory.SwitchToMainThreadAsync(package.DisposalToken);

            OleMenuCommandService commandService = await package.GetServiceAsync<IMenuCommandService, OleMenuCommandService>();
            Instance = new AnalyzeCodeCommand(package, commandService);
        }}

        private void Execute(object sender, EventArgs e)
        {{
            ThreadHelper.ThrowIfNotOnUIThread();

            string message = "AI Analysis started for {project_info.get('framework', 'your')} project...\n\n" +
                           "Analyzing {project_info.get('language', 'code')} patterns and architecture.";

            VsShellUtilities.ShowMessageBox(
                this.package,
                message,
                "AI Code Assistant",
                OLEMSGICON.OLEMSGICON_INFO,
                OLEMSGBUTTON.OLEMSGBUTTON_OK,
                OLEMSGDEFBUTTON.OLEMSGDEFBUTTON_FIRST);
        }}
    }}

    /// <summary>
    /// Generate Code Command specialized for {project_info.get('framework', 'development')}
    /// </summary>
    internal sealed class GenerateCodeCommand
    {{
        public const int CommandId = 0x0101;
        public static readonly Guid CommandSet = new Guid("a1b2c3d4-e5f6-7890-abcd-ef1234567891");

        private readonly AsyncPackage package;

        private GenerateCodeCommand(AsyncPackage package, OleMenuCommandService commandService)
        {{
            this.package = package;

            var menuCommandID = new CommandID(CommandSet, CommandId);
            var menuItem = new MenuCommand(this.Execute, menuCommandID);
            commandService.AddCommand(menuItem);
        }}

        public static GenerateCodeCommand Instance {{ get; private set; }}

        public static async Task InitializeAsync(AsyncPackage package)
        {{
            await ThreadHelper.JoinableTaskFactory.SwitchToMainThreadAsync(package.DisposalToken);

            OleMenuCommandService commandService = await package.GetServiceAsync<IMenuCommandService, OleMenuCommandService>();
            Instance = new GenerateCodeCommand(package, commandService);
        }}

        private void Execute(object sender, EventArgs e)
        {{
            ThreadHelper.ThrowIfNotOnUIThread();

            string message = "AI Code Generation for {project_info.get('language', 'multi-language')} {project_info.get('framework', '')} projects\n\n" +
                           "Ready to generate optimized code following best practices.";

            VsShellUtilities.ShowMessageBox(
                this.package,
                message,
                "AI Code Generator",
                OLEMSGICON.OLEMSGICON_INFO,
                OLEMSGBUTTON.OLEMSGBUTTON_OK,
                OLEMSGDEFBUTTON.OLEMSGDEFBUTTON_FIRST);
        }}
    }}

    /// <summary>
    /// Explain Code Command
    /// </summary>
    internal sealed class ExplainCodeCommand
    {{
        public const int CommandId = 0x0102;
        public static readonly Guid CommandSet = new Guid("a1b2c3d4-e5f6-7890-abcd-ef1234567891");

        private readonly AsyncPackage package;

        private ExplainCodeCommand(AsyncPackage package, OleMenuCommandService commandService)
        {{
            this.package = package;

            var menuCommandID = new CommandID(CommandSet, CommandId);
            var menuItem = new MenuCommand(this.Execute, menuCommandID);
            commandService.AddCommand(menuItem);
        }}

        public static ExplainCodeCommand Instance {{ get; private set; }}

        public static async Task InitializeAsync(AsyncPackage package)
        {{
            await ThreadHelper.JoinableTaskFactory.SwitchToMainThreadAsync(package.DisposalToken);

            OleMenuCommandService commandService = await package.GetServiceAsync<IMenuCommandService, OleMenuCommandService>();
            Instance = new ExplainCodeCommand(package, commandService);
        }}

        private void Execute(object sender, EventArgs e)
        {{
            ThreadHelper.ThrowIfNotOnUIThread();

            VsShellUtilities.ShowMessageBox(
                this.package,
                "AI Code Explanation ready!\n\nSelect code and use this command to get detailed explanations.",
                "AI Code Explainer", 
                OLEMSGICON.OLEMSGICON_INFO,
                OLEMSGBUTTON.OLEMSGBUTTON_OK,
                OLEMSGDEFBUTTON.OLEMSGDEFBUTTON_FIRST);
        }}
    }}
}}"""

        return extension_code

    def _generate_lsp_configuration(self, project_config: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация конфигурации LSP-AI"""

        project_info = project_config.get("project", {})

        return {
            "memory": {
                "file_store": {}
            },
            "models": {
                "completion_model": {
                    "type": "ollama",
                    "model": "qwen2.5-coder:7b",
                    "url": "http://localhost:11434",
                    "auth_token": None
                },
                "chat_model": {
                    "type": "ollama", 
                    "model": "qwen2.5-coder:7b",
                    "url": "http://localhost:11434",
                    "auth_token": None
                }
            },
            "completion": {
                "model": "completion_model",
                "parameters": {
                    "max_context": 2048,
                    "max_tokens": 128,
                    "fim": {
                        "enabled": True,
                        "prefix": "<fim_prefix>",
                        "middle": "<fim_middle>", 
                        "suffix": "<fim_suffix>"
                    },
                    "project_context": {
                        "language": project_info.get("language", ""),
                        "framework": project_info.get("framework", ""),
                        "project_type": project_info.get("type", "")
                    }
                }
            },
            "chat": {
                "model": "chat_model",
                "parameters": {
                    "max_context": 4096,
                    "max_tokens": 512,
                    "system_prompt": f"You are an expert {project_info.get('language', '')} developer specializing in {project_info.get('framework', '')} {project_info.get('type', '')} projects. Provide accurate, efficient, and well-documented code solutions following best practices."
                }
            }
        }

    def _generate_project_file(self, extension_name: str) -> str:
        """Генерация файла проекта .csproj"""

        project_content = f"""<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net48</TargetFramework>
    <RootNamespace>AICodeAssistant</RootNamespace>
    <AssemblyName>{extension_name}</AssemblyName>
    <GeneratePkgDefFile>true</GeneratePkgDefFile>
    <UseCodebase>true</UseCodebase>
    <IncludeAssemblyInVSIXContainer>true</IncludeAssemblyInVSIXContainer>
    <IncludeDebugSymbolsInVSIXContainer>false</IncludeDebugSymbolsInVSIXContainer>
    <IncludeDebugSymbolsInLocalVSIXDeployment>false</IncludeDebugSymbolsInLocalVSIXDeployment>
    <CopyBuildOutputToOutputDirectory>true</CopyBuildOutputToOutputDirectory>
    <CopyOutputSymbolsToOutputDirectory>false</CopyOutputSymbolsToOutputDirectory>
    <StartAction>Program</StartAction>
    <StartProgram Condition="'$(DevEnvDir)' != ''">$(DevEnvDir)devenv.exe</StartProgram>
    <StartArguments>/rootsuffix Exp</StartArguments>
  </PropertyGroup>

  <ItemGroup>
    <Compile Include="Properties\AssemblyInfo.cs" />
    <Compile Include="{extension_name}Package.cs" />
  </ItemGroup>

  <ItemGroup>
    <None Include="source.extension.vsixmanifest">
      <SubType>Designer</SubType>
    </None>
  </ItemGroup>

  <ItemGroup>
    <Reference Include="System" />
    <Reference Include="System.Data" />
    <Reference Include="System.Design" />
    <Reference Include="System.Drawing" />
    <Reference Include="System.Windows.Forms" />
    <Reference Include="System.Xml" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.VisualStudio.SDK" Version="17.0.0" ExcludeAssets="runtime">
      <IncludeAssets>compile; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>
    </PackageReference>
    <PackageReference Include="Microsoft.VisualStudio.LanguageServer.Client" Version="17.0.0" />
    <PackageReference Include="Microsoft.VSSDK.BuildTools" Version="17.0.0">
      <IncludeAssets>runtime; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>
      <PrivateAssets>all</PrivateAssets>
    </PackageReference>
  </ItemGroup>

  <ItemGroup>
    <VSCTCompile Include="AICodeAssistantPackage.vsct">
      <ResourceName>Menus.ctmenu</ResourceName>
    </VSCTCompile>
  </ItemGroup>

  <ItemGroup>
    <Content Include="lsp-config.json">
      <CopyToOutputDirectory>Always</CopyToOutputDirectory>
      <IncludeInVSIX>true</IncludeInVSIX>
    </Content>
  </ItemGroup>

</Project>"""

        return project_content

    def _save_extension_files(self, extension_name: str, files: Dict[str, str]):
        """Сохранение файлов расширения"""
        base_path = self.output_path / extension_name

        for filename, content in files.items():
            file_path = base_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"📝 Создан файл: {filename}")

    def _create_build_script(self, extension_name: str) -> str:
        """Создание скрипта сборки"""
        base_path = self.output_path / extension_name

        build_script_content = f"""@echo off
echo 🔨 Building Visual Studio Extension: {extension_name}

REM Check for MSBuild
where msbuild >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ MSBuild not found. Please install Visual Studio Build Tools.
    pause
    exit /b 1
)

REM Check for LSP-AI
where lsp-ai >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ LSP-AI not found. Installing...
    cargo install lsp-ai
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to install LSP-AI
        pause
        exit /b 1
    )
)

REM Build project
echo 🔨 Building project...
msbuild {extension_name}.csproj /p:Configuration=Release /p:Platform="Any CPU"

if %ERRORLEVEL% EQU 0 (
    echo ✅ Build completed successfully!
    echo 📦 VSIX file: bin\Release\{extension_name}.vsix

    REM Optional installation
    set /p install="Install extension in Visual Studio? (y/n): "
    if /i "%install%"=="y" (
        echo 🚀 Installing extension...
        "%%ProgramFiles%%\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\VSIXInstaller.exe" /q bin\Release\{extension_name}.vsix
        if %ERRORLEVEL% EQU 0 (
            echo ✅ Extension installed successfully!
        ) else (
            echo ⚠️ Installation may have failed. Try installing manually.
        )
    )
) else (
    echo ❌ Build failed
    pause
)

echo.
echo 💡 To use the extension:
echo    1. Start Visual Studio
echo    2. Open your project
echo    3. Look for "AI Assistant" in the menu
echo.
pause"""

        build_path = base_path / "build.bat"
        with open(build_path, 'w', encoding='utf-8') as f:
            f.write(build_script_content)

        print(f"🔨 Build script created: build.bat")
        return str(build_path)

# Простая демонстрация создания расширения
def create_sample_extension():
    """Создание примера расширения"""

    # Пример конфигурации проекта
    sample_config = {
        "project": {
            "name": "MyProject",
            "language": "Python", 
            "framework": "Django",
            "type": "Web Application"
        }
    }

    # Создание генератора
    generator = VisualStudioExtensionGenerator()

    # Генерация расширения
    extension_path = generator.generate_extension(sample_config, "AICodeAssistant")

    return f"✅ Sample Visual Studio extension created at: {extension_path}"

# Основная функция
def main():
    return create_sample_extension()

if __name__ == "__main__":
    result = main()
    print(result)
