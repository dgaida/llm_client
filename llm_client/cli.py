"""Command-line interface for LLM Client.

This module provides a rich CLI for interacting with various LLM providers,
including interactive chat, one-shot queries, token counting, and configuration.

Examples:
    # Quick query
    llm-client chat "What is Python?"

    # Interactive mode
    llm-client interactive --provider openai

    # Count tokens
    llm-client tokens "Hello world"

    # Stream response
    llm-client chat "Tell me a story" --stream
"""

import sys
from pathlib import Path

import click

from .config import LLMConfig, generate_config_template
from .llm_client import LLMClient
from .provider_factory import ProviderFactory
from .token_counter import TokenCounter

# Optional rich support for better formatting
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt

    # from rich.syntax import Syntax
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def echo(message: str, style: str | None = None) -> None:
    """Print message with optional rich formatting.

    Args:
        message: Message to print.
        style: Rich style (only if rich is available).
    """
    if RICH_AVAILABLE and console:
        console.print(message, style=style)
    else:
        click.echo(message)


def echo_error(message: str) -> None:
    """Print error message."""
    if RICH_AVAILABLE and console:
        console.print(f"[bold red]Error:[/bold red] {message}")
    else:
        click.echo(f"Error: {message}", err=True)


def echo_success(message: str) -> None:
    """Print success message."""
    if RICH_AVAILABLE and console:
        console.print(f"[bold green]✓[/bold green] {message}")
    else:
        click.echo(f"✓ {message}")


@click.group()
@click.version_option(version="0.3.0", prog_name="llm-client")
def cli() -> None:
    """Universal CLI for LLM access with multiple providers.

    Supports OpenAI, Groq, Google Gemini, and Ollama (local/cloud).

    \b
    Examples:
        llm-client chat "What is Python?"
        llm-client interactive --provider openai
        llm-client tokens "Hello world"
        llm-client config generate
    """
    pass


@cli.command()
@click.argument("prompt")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["auto", "openai", "groq", "gemini", "ollama"]),
    default="auto",
    help="LLM provider to use (default: auto-detect)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Specific model to use (overrides provider default)",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.7,
    help="Sampling temperature (0.0-2.0)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=512,
    help="Maximum tokens to generate",
)
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Stream response in real-time",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Load configuration from file",
)
@click.option(
    "--markdown/--no-markdown",
    default=True,
    help="Render response as markdown (requires rich)",
)
def chat(
    prompt: str,
    provider: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    stream: bool,
    config: str | None,
    markdown: bool,
) -> None:
    """Execute a one-shot chat completion.

    \b
    Examples:
        llm-client chat "Explain quantum computing"
        llm-client chat "Write a poem" --stream
        llm-client chat "Hello" --provider groq --model llama-3.3-70b-versatile
    """
    try:
        # Create client
        if config:
            client = LLMClient.from_config(
                config, provider=provider if provider != "auto" else None
            )
        else:
            client = LLMClient(
                api_choice=provider if provider != "auto" else None,
                llm=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Display info
        if RICH_AVAILABLE and console:
            console.print(f"[dim]Using: {client.api_choice} - {client.llm}[/dim]")
        else:
            click.echo(f"Using: {client.api_choice} - {client.llm}")

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Execute request
        if stream:
            if RICH_AVAILABLE and console:
                console.print("\n[bold]Response:[/bold]")
            else:
                click.echo("\nResponse:")

            for chunk in client.chat_completion_stream(messages):
                click.echo(chunk, nl=False)
            click.echo()  # New line at end
        else:
            response = client.chat_completion(messages)

            if RICH_AVAILABLE and console and markdown:
                console.print("\n[bold]Response:[/bold]")
                console.print(Markdown(response))
            else:
                click.echo("\nResponse:")
                click.echo(response)

    except Exception as e:
        echo_error(str(e))
        sys.exit(1)


@cli.command()
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["auto", "openai", "groq", "gemini", "ollama"]),
    default="auto",
    help="LLM provider to use",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Specific model to use",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.7,
    help="Sampling temperature",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Load configuration from file",
)
@click.option(
    "--system",
    "-s",
    default=None,
    help="System message to set context",
)
def interactive(
    provider: str,
    model: str | None,
    temperature: float,
    config: str | None,
    system: str | None,
) -> None:
    """Start an interactive chat session.

    Type 'exit', 'quit', or press Ctrl+C to exit.
    Type 'clear' to clear conversation history.
    Type 'switch <provider>' to change provider.

    \b
    Examples:
        llm-client interactive
        llm-client interactive --provider openai --system "You are a helpful coding assistant"
    """
    try:
        # Create client
        if config:
            client = LLMClient.from_config(
                config, provider=provider if provider != "auto" else None
            )
        else:
            client = LLMClient(
                api_choice=provider if provider != "auto" else None,
                llm=model,
                temperature=temperature,
            )

        # Initialize conversation
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        # Welcome message
        if RICH_AVAILABLE and console:
            console.print(
                Panel.fit(
                    f"[bold]Interactive Chat Mode[/bold]\n"
                    f"Provider: {client.api_choice}\n"
                    f"Model: {client.llm}\n"
                    f"Temperature: {temperature}\n\n"
                    f"Commands: exit, quit, clear, switch <provider>",
                    border_style="blue",
                )
            )
        else:
            click.echo("=== Interactive Chat Mode ===")
            click.echo(f"Provider: {client.api_choice}")
            click.echo(f"Model: {client.llm}")
            click.echo(f"Temperature: {temperature}")
            click.echo("\nCommands: exit, quit, clear, switch <provider>\n")

        # Chat loop
        while True:
            try:
                # Get user input
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                else:
                    user_input = click.prompt("\nYou", prompt_suffix=": ")

                # Handle commands
                if user_input.lower() in ["exit", "quit"]:
                    echo_success("Goodbye!")
                    break

                if user_input.lower() == "clear":
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    echo_success("Conversation cleared")
                    continue

                if user_input.lower().startswith("switch "):
                    new_provider = user_input.split()[1]
                    try:
                        client.switch_provider(new_provider)
                        echo_success(f"Switched to {new_provider} - {client.llm}")
                    except Exception as e:
                        echo_error(f"Could not switch provider: {e}")
                    continue

                # Add user message
                messages.append({"role": "user", "content": user_input})

                # Get response
                if RICH_AVAILABLE and console:
                    console.print("\n[bold green]Assistant[/bold green]:")
                else:
                    click.echo("\nAssistant:")

                response = client.chat_completion(messages)

                # Display response
                if RICH_AVAILABLE and console:
                    console.print(Markdown(response))
                else:
                    click.echo(response)

                # Add assistant message to history
                messages.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                echo("\n")
                echo_success("Goodbye!")
                break
            except EOFError:
                echo("\n")
                echo_success("Goodbye!")
                break

    except Exception as e:
        echo_error(str(e))
        sys.exit(1)


@cli.command()
@click.argument("text")
@click.option(
    "--model",
    "-m",
    default="gpt-4o-mini",
    help="Model to use for token encoding",
)
def tokens(text: str, model: str) -> None:
    """Count tokens in text using tiktoken.

    \b
    Examples:
        llm-client tokens "Hello world"
        llm-client tokens "Long text..." --model gpt-4o
    """
    try:
        counter = TokenCounter()
        count = counter.count_string_tokens(text, model=model)

        if RICH_AVAILABLE and console:
            console.print(f"[bold]Text:[/bold] {text[:50]}...")
            console.print(f"[bold]Model:[/bold] {model}")
            console.print(f"[bold green]Tokens:[/bold green] {count}")
        else:
            click.echo(f"Text: {text[:50]}...")
            click.echo(f"Model: {model}")
            click.echo(f"Tokens: {count}")

    except Exception as e:
        echo_error(str(e))
        sys.exit(1)


@cli.group()
def config() -> None:
    """Manage configuration files."""
    pass


@config.command(name="generate")
@click.option(
    "--output",
    "-o",
    default="llm_config.yaml",
    help="Output file path",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format",
)
def generate_config(output: str, format: str) -> None:
    """Generate a template configuration file.

    \b
    Examples:
        llm-client config generate
        llm-client config generate --output my_config.json --format json
    """
    try:
        generate_config_template(output, format=format)
        echo_success(f"Configuration template generated: {output}")
    except Exception as e:
        echo_error(str(e))
        sys.exit(1)


@config.command(name="validate")
@click.argument("config_file", type=click.Path(exists=True))
def validate_config(config_file: str) -> None:
    """Validate a configuration file.

    \b
    Examples:
        llm-client config validate llm_config.yaml
    """
    try:
        config = LLMConfig.from_file(config_file)
        is_valid, errors = config.validate()

        if is_valid:
            echo_success(f"Configuration is valid: {config_file}")

            if RICH_AVAILABLE and console:
                console.print(f"\n[bold]Default Provider:[/bold] {config.default_provider}")
                console.print(
                    f"[bold]Available Providers:[/bold] {', '.join(config.list_providers())}"
                )
        else:
            echo_error(f"Configuration is invalid: {config_file}")
            for error in errors:
                echo(f"  - {error}", style="red")
            sys.exit(1)
    except Exception as e:
        echo_error(str(e))
        sys.exit(1)


@config.command(name="show")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--provider",
    "-p",
    default=None,
    help="Show specific provider configuration",
)
def show_config(config_file: str, provider: str | None) -> None:
    """Display configuration details.

    \b
    Examples:
        llm-client config show llm_config.yaml
        llm-client config show llm_config.yaml --provider openai
    """
    try:
        config = LLMConfig.from_file(config_file)

        if RICH_AVAILABLE and console:
            if provider:
                # Show specific provider
                provider_config = config.get_provider_config(provider)

                table = Table(title=f"Provider: {provider}")
                table.add_column("Setting", style="cyan")
                table.add_column("Value", style="green")

                for key, value in provider_config.items():
                    table.add_row(key, str(value))

                console.print(table)
            else:
                # Show all providers
                table = Table(title="LLM Configuration")
                table.add_column("Provider", style="cyan")
                table.add_column("Model", style="green")
                table.add_column("Temperature", style="yellow")
                table.add_column("Max Tokens", style="magenta")

                for prov in config.list_providers():
                    prov_config = config.get_provider_config(prov)
                    table.add_row(
                        f"{'* ' if prov == config.default_provider else '  '}{prov}",
                        prov_config.get("model", "N/A"),
                        str(prov_config.get("temperature", "N/A")),
                        str(prov_config.get("max_tokens", "N/A")),
                    )

                console.print(table)
                console.print(f"\n[dim]* Default provider: {config.default_provider}[/dim]")
        else:
            # Plain text output
            if provider:
                click.echo(f"Provider: {provider}")
                provider_config = config.get_provider_config(provider)
                for key, value in provider_config.items():
                    click.echo(f"  {key}: {value}")
            else:
                click.echo("LLM Configuration:")
                click.echo(f"Default: {config.default_provider}\n")
                for prov in config.list_providers():
                    prov_config = config.get_provider_config(prov)
                    marker = "*" if prov == config.default_provider else " "
                    click.echo(f"{marker} {prov}:")
                    click.echo(f"    Model: {prov_config.get('model', 'N/A')}")
                    click.echo(f"    Temperature: {prov_config.get('temperature', 'N/A')}")
                    click.echo(f"    Max Tokens: {prov_config.get('max_tokens', 'N/A')}")
    except Exception as e:
        echo_error(str(e))
        sys.exit(1)


@cli.command()
def providers() -> None:
    """List available LLM providers and their status.

    \b
    Examples:
        llm-client providers
    """
    try:
        all_providers = ["openai", "groq", "gemini", "ollama"]
        available = ProviderFactory.get_available_providers()

        if RICH_AVAILABLE and console:
            table = Table(title="Available LLM Providers")
            table.add_column("Provider", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Package", style="yellow")

            for prov in all_providers:
                status = "✓ Available" if prov in available else "✗ Not installed"
                package = {
                    "openai": "openai",
                    "groq": "groq",
                    "gemini": "openai",
                    "ollama": "ollama",
                }[prov]
                style = "green" if prov in available else "red"
                table.add_row(prov, f"[{style}]{status}[/{style}]", package)

            console.print(table)
        else:
            click.echo("Available LLM Providers:")
            for prov in all_providers:
                status = "✓" if prov in available else "✗"
                click.echo(f"  {status} {prov}")

    except Exception as e:
        echo_error(str(e))
        sys.exit(1)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["auto", "openai", "groq", "gemini", "ollama"]),
    default="auto",
    help="LLM provider to use",
)
@click.option(
    "--system",
    "-s",
    default="You are a helpful assistant that analyzes files.",
    help="System message",
)
def analyze(file_path: str, provider: str, system: str) -> None:
    """Analyze a file using LLM.

    \b
    Examples:
        llm-client analyze code.py
        llm-client analyze document.txt --provider openai
    """
    try:
        # Read file
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        # Create client
        client = LLMClient(api_choice=provider if provider != "auto" else None)

        # Prepare prompt
        prompt = f"Please analyze this file:\n\nFilename: {path.name}\n\n```\n{content}\n```"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Show info
        echo(f"Analyzing: {file_path}")
        echo(f"Provider: {client.api_choice} - {client.llm}\n")

        # Get response
        response = client.chat_completion(messages)

        # Display
        if RICH_AVAILABLE and console:
            console.print(Markdown(response))
        else:
            click.echo(response)

    except Exception as e:
        echo_error(str(e))
        sys.exit(1)


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
