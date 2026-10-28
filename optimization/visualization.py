from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gantt(problem, result, path=None):
    if not result.get("schedule"):
        raise ValueError("No feasible schedule to plot")
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = {j.id: plt.get_cmap("tab20")(i % 20) for i, j in enumerate(problem.jobs)}
    for row in result["schedule"]:
        ax.barh(
            row["machine"],
            row["end"] - row["start"],
            left=row["start"],
            height=0.7,
            color=colors[row["job"]],
            edgecolor="white",
        )
        ax.text(
            (row["start"] + row["end"]) / 2,
            row["machine"],
            f"{row['job']}.{row['operation']}",
            ha="center",
            va="center",
            fontsize=8,
        )
    for machine, slots in problem.blackouts.items():
        for start, end in slots:
            ax.barh(
                machine,
                end - start,
                left=start,
                height=0.9,
                color="none",
                edgecolor="black",
                hatch="///",
            )
    ax.set_yticks(
        range(problem.machines), [f"Machine {i}" for i in range(problem.machines)]
    )
    ax.set_xlabel("Integer time units")
    ax.set_title(f"{problem.name}: {result['status']}, makespan {result['makespan']}")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
