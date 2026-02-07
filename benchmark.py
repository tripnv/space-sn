from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import plotly.graph_objects as go

from core.engine import reset, step
from agents.search import BFSAgent, DFSAgent, BestFirstAgent, AStarAgent
from agents.jax_search import JAXBFSAgent, JAXDFSAgent, JAXBestFirstAgent, JAXAStarAgent

AGENTS = {
    "BFS": BFSAgent,
    "DFS": DFSAgent,
    "Best-First": BestFirstAgent,
    "A*": AStarAgent,
    "JAX-BFS": JAXBFSAgent,
    "JAX-DFS": JAXDFSAgent,
    "JAX-Best-First": JAXBestFirstAgent,
    "JAX-A*": JAXAStarAgent,
}

COLORS = {
    "BFS": "rgba(31,119,180,{a})",
    "DFS": "rgba(255,127,14,{a})",
    "Best-First": "rgba(44,160,44,{a})",
    "A*": "rgba(214,39,40,{a})",
    "JAX-BFS": "rgba(31,119,180,{a})",
    "JAX-DFS": "rgba(255,127,14,{a})",
    "JAX-Best-First": "rgba(44,160,44,{a})",
    "JAX-A*": "rgba(214,39,40,{a})",
}


def _run_episode(agent_name: str, grid_num: int, max_steps: int, seed: int) -> tuple[str, dict]:
    max_length = grid_num ** 3
    reset_jit = jax.jit(reset, static_argnums=(1, 2))
    step_jit = jax.jit(step, static_argnums=(2,))

    key = jax.random.PRNGKey(seed)
    state = reset_jit(key, grid_num, max_length)
    agent_obj = AGENTS[agent_name](grid_num=grid_num)

    t0 = time.perf_counter()
    steps = 0
    while bool(state.alive) and steps < max_steps:
        action = agent_obj.act(state)
        state = step_jit(state, jnp.int32(action), grid_num)
        steps += 1
    elapsed = time.perf_counter() - t0

    return agent_name, {"score": int(state.score), "steps": steps, "time": elapsed}


def _print_progress(counts: dict[str, int], total: int):
    parts = [f"{name}: {counts[name]}/{total}" for name in counts]
    line = "  " + "  |  ".join(parts)
    sys.stderr.write(f"\r{line}")
    sys.stderr.flush()


def _make_bar_charts(results: list[dict], out_dir: Path):
    import numpy as np

    metrics = [("scores", "Score (food eaten)"), ("steps", "Steps"), ("times", "Time (s)")]
    for key, label in metrics:
        fig = go.Figure()
        for r in results:
            vals = np.array(r[key])
            fig.add_trace(go.Bar(
                name=r["agent"],
                x=[r["agent"]],
                y=[vals.mean()],
                error_y=dict(type="data", array=[vals.std() / np.sqrt(len(vals))]),
                marker_color=COLORS[r["agent"]].format(a=0.8),
            ))
        fig.update_layout(
            title=f"Mean {label} (± SE)",
            yaxis_title=label,
            showlegend=False,
            template="plotly_white",
        )
        fig.write_html(out_dir / f"bar_{key}.html")


def _make_histograms(results: list[dict], out_dir: Path):
    metrics = [("scores", "Score (food eaten)"), ("steps", "Steps"), ("times", "Time (s)")]
    for key, label in metrics:
        fig = go.Figure()
        for r in results:
            fig.add_trace(go.Histogram(
                x=r[key],
                name=r["agent"],
                opacity=0.5,
                marker_color=COLORS[r["agent"]].format(a=0.5),
            ))
        fig.update_layout(
            barmode="overlay",
            title=f"Distribution of {label}",
            xaxis_title=label,
            yaxis_title="Count",
            template="plotly_white",
        )
        fig.write_html(out_dir / f"hist_{key}.html")


def _make_ecdfs(results: list[dict], out_dir: Path):
    import numpy as np

    metrics = [("scores", "Score (food eaten)"), ("steps", "Steps"), ("times", "Time (s)")]
    for key, label in metrics:
        fig = go.Figure()
        for r in results:
            sorted_vals = np.sort(r[key])
            ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            fig.add_trace(go.Scatter(
                x=sorted_vals,
                y=ecdf,
                mode="lines",
                name=r["agent"],
                line=dict(color=COLORS[r["agent"]].format(a=1.0)),
            ))
        fig.update_layout(
            title=f"ECDF of {label}",
            xaxis_title=label,
            yaxis_title="Cumulative Probability",
            template="plotly_white",
        )
        fig.write_html(out_dir / f"ecdf_{key}.html")


@click.command()
@click.option("--episodes", "-n", type=int, default=1000, help="Episodes per algorithm.")
@click.option("--grid-num", "-g", type=int, default=10, help="Grid size per axis.")
@click.option("--max-steps", type=int, default=10_000, help="Max steps per episode.")
@click.option("--seed", "-s", type=int, default=0, help="Base random seed.")
@click.option("--out", "-o", type=click.Path(), default="benchmarks/", help="Output directory for plots.")
@click.option("--agents", "-a", type=str, default=None, help="Comma-separated agent names (default: all).")
def benchmark(episodes: int, grid_num: int, max_steps: int, seed: int, out: str, agents: str | None):
    """Run Monte Carlo benchmark across all search agents."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if agents:
        agent_names = [a.strip() for a in agents.split(",")]
        for name in agent_names:
            if name not in AGENTS:
                raise click.BadParameter(f"Unknown agent: {name!r}. Choose from: {', '.join(AGENTS)}")
    else:
        agent_names = list(AGENTS.keys())
    click.echo(f"Running {episodes} episodes per agent | grid: {grid_num} | max_steps: {max_steps} | seed: {seed}")
    click.echo(f"Agents: {', '.join(agent_names)}\n")

    # Submit all individual episodes across all agents
    collected: dict[str, dict] = {
        name: {"agent": name, "scores": [], "steps": [], "times": []}
        for name in agent_names
    }
    counts = {name: 0 for name in agent_names}

    with ProcessPoolExecutor() as executor:
        futures = []
        for name in agent_names:
            for i in range(episodes):
                futures.append(
                    executor.submit(_run_episode, name, grid_num, max_steps, seed + i)
                )

        _print_progress(counts, episodes)
        for future in as_completed(futures):
            agent_name, result = future.result()
            collected[agent_name]["scores"].append(result["score"])
            collected[agent_name]["steps"].append(result["steps"])
            collected[agent_name]["times"].append(result["time"])
            counts[agent_name] += 1
            _print_progress(counts, episodes)

    sys.stderr.write("\n")
    results = [collected[name] for name in agent_names]

    click.echo("\n--- Summary ---")
    import numpy as np
    for r in results:
        s, st, t = np.array(r["scores"]), np.array(r["steps"]), np.array(r["times"])
        click.echo(
            f"{r['agent']:>12s}  score: {s.mean():.1f}±{s.std():.1f}  "
            f"steps: {st.mean():.0f}±{st.std():.0f}  "
            f"time: {t.mean():.3f}±{t.std():.3f}s"
        )

    click.echo(f"\nGenerating plots in {out_dir}/ ...")
    _make_bar_charts(results, out_dir)
    _make_histograms(results, out_dir)
    _make_ecdfs(results, out_dir)
    click.echo("Done. Generated 9 HTML plots:")
    for f in sorted(out_dir.glob("*.html")):
        click.echo(f"  {f}")


if __name__ == "__main__":
    benchmark()
