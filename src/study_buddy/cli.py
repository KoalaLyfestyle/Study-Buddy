"""Command Line Interface for Study Buddy RAG Assistant."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from study_buddy.rag_assistant import RAGAssistant
from study_buddy.config import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()


@click.group()
@click.option('--model', default=None, help='Ollama model to use (e.g., mistral, llama2)')
@click.pass_context
def cli(ctx, model):
    """Study Buddy RAG Assistant - Offline AI-powered study companion."""
    ctx.ensure_object(dict)
    
    try:
        ctx.obj['assistant'] = RAGAssistant(model_name=model)
        if model:
            console.print(f"[green]Using model: {model}[/green]")
    except Exception as e:
        console.print(f"[red]Error initializing RAG Assistant: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('path', type=click.Path(exists=True, path_type=Path))
@click.pass_context
def ingest(ctx, path):
    """Ingest documents from a file or directory."""
    assistant = ctx.obj['assistant']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        if path.is_file():
            task = progress.add_task(f"Ingesting {path.name}...", total=1)
            success = assistant.ingest_document(path)
            progress.update(task, completed=1)
            
            if success:
                console.print(f"[green]✓ Successfully ingested {path.name}[/green]")
            else:
                console.print(f"[red]✗ Failed to ingest {path.name}[/red]")
                
        elif path.is_dir():
            task = progress.add_task(f"Ingesting documents from {path.name}...", total=None)
            results = assistant.ingest_directory(path)
            progress.update(task, completed=1)
            
            # Display results
            table = Table(title="Ingestion Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="green")
            
            table.add_row("Total Files", str(results["total_files"]))
            table.add_row("Successful", str(results["successful"]))
            table.add_row("Failed", str(results["failed"]))
            
            console.print(table)
            
            if results["failed_files"]:
                console.print("\n[yellow]Failed files:[/yellow]")
                for file in results["failed_files"]:
                    console.print(f"  • {file}")


@cli.command()
@click.argument('question')
@click.option('--max-results', '-n', default=None, type=int, help='Maximum number of context documents')
@click.pass_context
def query(ctx, question, max_results):
    """Ask a question to the RAG assistant."""
    assistant = ctx.obj['assistant']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating response...", total=None)
        response = assistant.query(question, max_results)
        progress.update(task, completed=1)
    
    # Display the response
    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan] {response['question']}",
        title="Query"
    ))
    
    if "error" in response:
        console.print(Panel(
            f"[red]{response['answer']}[/red]",
            title="Error"
        ))
    else:
        console.print(Panel(
            response["answer"],
            title="Answer"
        ))
        
        # Show context sources
        if response.get("context_documents"):
            console.print("\n[bold yellow]Sources used:[/bold yellow]")
            for i, doc in enumerate(response["context_documents"], 1):
                source = doc.get("source", "Unknown")
                preview = doc.get("content", "")
                console.print(f"  {i}. {source}")
                console.print(f"     Preview: {preview[:100]}...")


@cli.command()
@click.argument('search_term')
@click.option('--max-results', '-n', default=None, type=int, help='Maximum number of results')
@click.pass_context
def search(ctx, search_term, max_results):
    """Search for documents containing specific terms."""
    assistant = ctx.obj['assistant']
    
    results = assistant.search_documents(search_term, max_results)
    
    if not results:
        console.print("[yellow]No documents found matching your search.[/yellow]")
        return
    
    console.print(f"\n[bold green]Found {len(results)} relevant documents:[/bold green]\n")
    
    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        source = result.get("source", "Unknown")
        content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
        
        console.print(Panel(
            f"[bold]Source:[/bold] {source}\n"
            f"[bold]Relevance Score:[/bold] {score:.4f}\n\n"
            f"{content_preview}",
            title=f"Result {i}"
        ))


@cli.command()
@click.pass_context
def status(ctx):
    """Show the current status of the knowledge base and system."""
    assistant = ctx.obj['assistant']
    
    info = assistant.get_knowledge_base_info()
    
    if "error" in info:
        console.print(f"[red]Error getting status: {info['error']}[/red]")
        return
    
    # Vector Store Info
    vector_info = info.get("vector_store", {})
    table = Table(title="Knowledge Base Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Documents Indexed", str(vector_info.get("document_count", 0)))
    table.add_row("Collection Name", vector_info.get("collection_name", "N/A"))
    table.add_row("Embedding Model", vector_info.get("embedding_model", "N/A"))
    
    console.print(table)
    
    # LLM Info
    llm_info = info.get("llm", {})
    llm_table = Table(title="LLM Status")
    llm_table.add_column("Property", style="cyan")
    llm_table.add_column("Value", style="green")
    
    ollama_status = "✓ Running" if llm_info.get("ollama_running") else "✗ Not Running"
    llm_table.add_row("Ollama Server", ollama_status)
    llm_table.add_row("Current Model", llm_info.get("current_model", "N/A"))
    
    available_models = llm_info.get("available_models", [])
    if available_models:
        llm_table.add_row("Available Models", ", ".join(available_models[:3]) + f" (+{len(available_models)-3} more)" if len(available_models) > 3 else ", ".join(available_models))
    else:
        llm_table.add_row("Available Models", "None (Ollama may not be running)")
    
    console.print(llm_table)


@cli.command()
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def clear(ctx, confirm):
    """Clear all documents from the knowledge base."""
    assistant = ctx.obj['assistant']
    
    if not confirm:
        if not Confirm.ask("Are you sure you want to clear all documents from the knowledge base?"):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Clearing knowledge base...", total=None)
        success = assistant.clear_knowledge_base()
        progress.update(task, completed=1)
    
    if success:
        console.print("[green]✓ Knowledge base cleared successfully.[/green]")
    else:
        console.print("[red]✗ Failed to clear knowledge base.[/red]")


@cli.command()
def interactive():
    """Start an interactive query session."""
    assistant = RAGAssistant()
    
    console.print(Panel(
        "[bold green]Study Buddy Interactive Mode[/bold green]\n\n"
        "Ask questions about your documents. Type 'quit' to exit.",
        title="Welcome"
    ))
    
    while True:
        try:
            question = console.input("\n[bold cyan]Your question:[/bold cyan] ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not question.strip():
                continue
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Thinking...", total=None)
                response = assistant.query(question)
                progress.update(task, completed=1)
            
            console.print(Panel(
                response["answer"],
                title="Answer"
            ))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


if __name__ == "__main__":
    cli()