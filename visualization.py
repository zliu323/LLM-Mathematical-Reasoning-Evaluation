import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

methods = ["Direct", "CoT", "Self-Refine"]

def load_model_results(summary_path, model_name):
    frame = pd.read_csv(summary_path)
    required = {"model", "dataset", "method", "accuracy", "avg_response_length_words"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {', '.join(sorted(missing))}")

    model_rows = frame[frame["model"] == model_name]
    data = {}
    for dataset in model_rows["dataset"].drop_duplicates():
        dataset_rows = model_rows[model_rows["dataset"] == dataset].set_index("method")
        data[dataset] = {
            "accuracy": [dataset_rows.at[method, "accuracy"] for method in methods],
            "length": [dataset_rows.at[method, "avg_response_length_words"] for method in methods],
        }
    return data

def get_global_max_length(*model_dicts):
    max_len = 0
    for model in model_dicts:
        for dataset in model.values():
            for v in dataset["length"]:
                if v is not None and v > max_len:
                    max_len = v
    return max_len

def make_combined(model_name, data, save_path, max_length_ylim):
    datasets = list(data.keys())
    x = list(range(len(datasets)))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # Accuracy
    for i, method in enumerate(methods):
        vals = [data[d]["accuracy"][i] if data[d]["accuracy"][i] is not None else float("nan") for d in datasets]
        pos = [p + (i - 1) * width for p in x]
        axes[0].bar(pos, vals, width=width, label=method)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Dataset")
    axes[0].set_title("Accuracy")

    # Response length
    for i, method in enumerate(methods):
        vals = [data[d]["length"][i] if data[d]["length"][i] is not None else float("nan") for d in datasets]
        pos = [p + (i - 1) * width for p in x]
        axes[1].bar(pos, vals, width=width, label=method)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylim(0, max_length_ylim)   # same range for both images
    axes[1].set_ylabel("Average Response Length (words)")
    axes[1].set_xlabel("Dataset")
    axes[1].set_title("Response Length")

    # Main title and legend
    fig.suptitle(f"{model_name} Results", fontsize=14, y=0.98)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.93))

    # leave enough room for title + legend
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    outdir = base_dir / "results"
    outdir.mkdir(exist_ok=True)

    summary_path = outdir / "summary.csv"
    qwen = load_model_results(summary_path, "Qwen/Qwen2.5-Math-1.5B")
    deepseek = load_model_results(
        summary_path, "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )

    global_max_length = get_global_max_length(qwen, deepseek)
    global_max_length = global_max_length * 1.1   # add a little top space

    make_combined("Qwen2.5-Math-1.5B", qwen, outdir / "qwen_results.png", global_max_length)
    make_combined("DeepSeek-R1-Distill-Qwen-1.5B", deepseek, outdir / "deepseek_results.png", global_max_length)

    print("Saved:")
    print(outdir / "qwen_results.png")
    print(outdir / "deepseek_results.png")
