from rich.console import Console
from rich.live import Live
from rich.table import Table

# Create a single instance of Console
console = Console()

live = None
stats_table = None


# Function to create an initial empty table or placeholder
def get_initial_renderable():
    if stats_table is None:
        table = Table(title="Starting Table")
        table.add_column("Status", justify="center")
        table.add_row("Initializing...")
    return stats_table


# Function to start the Live context
def start_live_context():
    global live
    live = Live(get_initial_renderable(), console=console, refresh_per_second=2)
    return live
