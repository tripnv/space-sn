import game
import click


# @click.group("run")
# @click.option(
#     "--headless/--render",
#     default=True,
#     help="Whether to display the game, or to run it headless",
# )
# def run(mode):
#     agent = init_agent()
#     environment = init_environment()
#     environment.agent = agent

#     if mode == "render":
#         environment.run_sketch()
#     else:
#         click.echo(f"The {agent.search_algorithm} scored: {environment.headless_run()}")


# @run.command()
# @click.option(
#     "-a",
#     "--agent",
#     default="human",
#     type=click.Choice(["human", "bfs"]),
#     help="Specify the agent type. Admissible values include 'human' and other search algorithms (e.g bfs)",
# )
# # @click.pass_context()
# def init_agent(agent):
#     if agent == "human":
#         return None
#     else:
#         return game.Agent(agent)


# @run.command()
# @click.option(
#     "-fr",
#     "--frame-rate",
#     default=60,
#     type=int,
#     help="Specify the frame rate of the displayed game. Defaults to 60",
# )
# def init_environment(fr):
#     environment = game.Environment()
#     environment.frame_rate = fr
#     return environment


# if __name__ == "__main__":
#     run()


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
