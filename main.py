from __future__ import annotations

import time

import click
import jax
import jax.numpy as jnp

from config import load_config
from core.engine import reset, step
from agents.search import BFSAgent, DFSAgent, BestFirstAgent, AStarAgent

AGENTS = {
    "bfs": BFSAgent,
    "dfs": DFSAgent,
    "best-first": BestFirstAgent,
    "astar": AStarAgent,
}


@click.command()
@click.option(
    "--agent", "-a",
    type=click.Choice(["bfs", "dfs", "best-first", "astar"]),
    default="bfs",
    help="Search agent type.",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["headless", "render"]),
    default="headless",
    help="Run headless (no window) or with 3D rendering.",
)
@click.option("--grid-num", "-g", type=int, default=None, help="Grid size per axis (default from config).")
@click.option("--seed", "-s", type=int, default=0, help="Random seed.")
@click.option("--max-steps", type=int, default=10_000, help="Max steps before stopping.")
@click.option("--record", "-r", type=click.Path(), default=None, help="Record to gif (e.g. samples/out.gif).")
@click.option("--record-frames", type=int, default=300, help="Number of frames to record.")
def run(agent: str, mode: str, grid_num: int | None, seed: int, max_steps: int, record: str | None, record_frames: int):
    """Space Snake — JAX-powered 3D snake with search agents."""
    cfg = load_config()
    grid_num = grid_num or cfg["environment"]["grid_num"]
    max_length = grid_num ** 3

    reset_jit = jax.jit(reset, static_argnums=(1, 2))
    step_jit = jax.jit(step, static_argnums=(2,))

    key = jax.random.PRNGKey(seed)
    state = reset_jit(key, grid_num, max_length)

    agent_obj = AGENTS[agent](grid_num=grid_num)
    click.echo(f"agent: {agent_obj.agent_type}  grid: {grid_num}  mode: {mode}")

    if record:
        _run_record(state, agent_obj, step_jit, grid_num, max_steps, agent_name=agent,
                     out_path=record, num_frames=record_frames)
    elif mode == "headless":
        _run_headless(state, agent_obj, step_jit, grid_num, max_steps)
    else:
        _run_render(state, agent_obj, step_jit, grid_num, max_steps, agent_name=agent)


def _run_headless(state, agent_obj, step_jit, grid_num, max_steps):
    t0 = time.time()
    steps = 0
    while bool(state.alive) and steps < max_steps:
        action = agent_obj.act(state)
        state = step_jit(state, jnp.int32(action), grid_num)
        steps += 1

    elapsed = time.time() - t0
    click.echo(
        f"score: {int(state.score)}  length: {int(state.length)}  "
        f"steps: {steps}  time: {elapsed:.2f}s"
    )


def _run_record(state, agent_obj, step_jit, grid_num, max_steps, agent_name="",
                 out_path="out.gif", num_frames=300):
    from pathlib import Path
    from PIL import Image
    from rendering.renderer import Renderer

    renderer = Renderer(grid_num=grid_num, agent_name=agent_name)
    frames: list[Image.Image] = []
    steps = 0

    for _ in range(num_frames):
        if not bool(state.alive) or steps >= max_steps:
            break
        action = agent_obj.act(state)
        state = step_jit(state, jnp.int32(action), grid_num)
        steps += 1
        renderer.update(state)
        snap = renderer.snapshot()
        if snap is not None:
            frames.append(Image.fromarray(snap))

    if frames:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:],
                        duration=33, loop=0, optimize=True)
        click.echo(f"saved {len(frames)} frames to {out}")
    else:
        click.echo("no frames captured")

    click.echo(f"score: {int(state.score)}  length: {int(state.length)}  steps: {steps}")


def _run_render(state, agent_obj, step_jit, grid_num, max_steps, agent_name=""):
    from rendering.renderer import Renderer
    from rendercanvas.auto import loop

    renderer = Renderer(grid_num=grid_num, agent_name=agent_name)
    renderer.update(state)

    frame_interval = 1.0 / 30  # target ~30 fps for visualization
    steps = 0

    def animate():
        nonlocal state, steps
        if not bool(state.alive) or steps >= max_steps:
            click.echo(
                f"done — score: {int(state.score)}  length: {int(state.length)}  steps: {steps}"
            )
            renderer.close()
            return

        action = agent_obj.act(state)
        state = step_jit(state, jnp.int32(action), grid_num)
        steps += 1
        renderer.update(state)

    renderer.canvas.request_draw(animate)
    loop.run()


if __name__ == "__main__":
    run()
