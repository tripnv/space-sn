import game
import click


@click.command()
@click.option(
    "--agent",
    "-a",
    type=click.Choice(["human", "bfs", "dfs", "best-first", "astar"]),
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
    "--frame-rate", "-fr", type=float, default=60, help="Frame rate for render mode"
)
def run(agent, mode, frame_rate):
    agent_ = game.Agent(agent)
    environment = game.Environment(agent_, frame_rate=frame_rate)
    click.echo(agent)
    if mode == "headless":
        score = environment.run_headless()
        click.echo(f"{agent_.agent_type} agent scored: {score}")
    elif mode == "render":
        environment.run_sketch()


if __name__ == "__main__":
    run()
