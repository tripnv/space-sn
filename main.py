import game
import click


@click.command()
@click.option(
    "--agent",
    "-a",
    type=click.Choice(["human", "bfs", "dfs"]),
    default="human",
    help="Agent type",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["headless", "render"]),
    default="headless",
    help="Operation mode",
)
@click.option(
    "--frame-rate", "-fr", type=int, default=60, help="Frame rate for render mode"
)
def run(agent, mode, frame_rate):
    agent_ = game.Agent(agent)
    environment = game.Environment(agent_)
    environment.frame_rate = frame_rate
    click.echo(agent)
    if mode == "headless":
        environment.run_headless()
    elif mode == "render":
        environment.run_sketch()


if __name__ == "__main__":
    run()
