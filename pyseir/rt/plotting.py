import matplotlib.pyplot as plt
from datetime import timedelta


def plot_smoothing(x, original, processed, timeseries_type) -> plt.Figure:
    """
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        x[-len(original) :],
        original,
        alpha=0.3,
        label=timeseries_type.value.replace("_", " ").title() + "Shifted",
    )
    ax.plot(x[-len(original) :], processed)  # TODO: Ask Alex why this isn't len(processed)
    plt.grid(True, which="both")
    plt.xticks(rotation=30)
    plt.xlim(min(x[-len(original) :]), max(x) + timedelta(days=2))

    return fig


def plot_posteriors(x) -> plt.Figure:
    """
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, alpha=0.1, color="k")
    plt.grid(alpha=0.4)
    plt.xlabel("$R_t$", fontsize=16)
    plt.title("Posteriors", fontsize=18)
    return fig


def plot_rt(df, include_deaths, shift_deaths, display_name) -> plt.Figure:
    """"""
    fig, ax = plt.subplots(figsize=(10, 6))

    if "Rt_ci5__new_deaths" in df:
        if include_deaths:
            plt.fill_between(
                df.index,
                df["Rt_ci5__new_deaths"],
                df["Rt_ci95__new_deaths"],
                alpha=0.2,
                color="firebrick",
            )
        # Show for reference even if not used
        plt.scatter(
            df.index,
            df["Rt_MAP__new_deaths"].shift(periods=shift_deaths),
            s=25,
            color="firebrick",
            label="New Deaths",
        )

    if "Rt_ci5__new_cases" in df:
        plt.fill_between(
            df.index,
            df["Rt_ci5__new_cases"],
            df["Rt_ci95__new_cases"],
            alpha=0.2,
            color="steelblue",
        )
        plt.scatter(
            df.index,
            df["Rt_MAP__new_cases"],
            alpha=1,
            s=25,
            color="steelblue",
            label="New Cases",
            marker="s",
        )

    if "Rt_MAP_composite" in df:
        plt.scatter(
            df.index,
            df["Rt_MAP_composite"],
            alpha=1,
            s=25,
            color="black",
            label="Inferred $R_{t}$ Web",
            marker="d",
        )

    if "Rt_ci95_composite" in df:
        plt.fill_between(
            df.index,
            df["Rt_ci95_composite"],
            2 * df["Rt_MAP_composite"] - df["Rt_ci95_composite"],
            alpha=0.2,
            color="gray",
        )

    plt.hlines([0.9], *plt.xlim(), alpha=1, color="g")
    plt.hlines([1.1], *plt.xlim(), alpha=1, color="gold")
    plt.hlines([1.4], *plt.xlim(), alpha=1, color="r")

    plt.xticks(rotation=30)
    plt.grid(True)
    plt.xlim(df.index.min() - timedelta(days=2), df.index.max() + timedelta(days=2))
    plt.ylim(0.0, 3.0)
    plt.ylabel("$R_t$", fontsize=16)
    plt.legend()
    plt.title(display_name, fontsize=14)

    return fig
