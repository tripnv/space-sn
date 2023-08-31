import game
import click
import py5_tools
import skvideo.io


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
        # writer = skvideo.io.FFmpegWriter(
        #     environment.output_folder_path + ".mp4",
        # )
        # py5_tools.offline_frame_processing(
        #     writer.writeFrame,
        #     batch_size=1,
        #     limit=0,
        #     sketch=environment,
        #     complete_func=writer.close,
        # )
        environment.run_sketch()


if __name__ == "__main__":
    run()
