import csv
import os

import matplotlib.pyplot as plt

script_directory = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_directory, "results.csv")
image_directory = os.path.join(script_directory, "images")
os.makedirs(image_directory, exist_ok=True)

with open(results_path, newline="") as results_file:
    results = list(csv.DictReader(results_file))

for row in results:
    row["epoch"] = int(row["epoch"])
    row["validation_accuracy"] = float(row["validation_accuracy"])
    row["epoch_time_seconds"] = float(row["epoch_time_seconds"])


def rows_for(experiment_ids):
    return [row for row in results if row["experiment_id"] in experiment_ids]


final_results = {}
for row in results:
    experiment_id = row["experiment_id"]
    if (experiment_id not in final_results
            or row["epoch"] > final_results[experiment_id]["epoch"]):
        final_results[experiment_id] = row


print("Training time summary:")
for experiment_id in final_results:
    experiment_rows = [
        row for row in results
        if row["experiment_id"] == experiment_id
    ]
    times = [row["epoch_time_seconds"] for row in experiment_rows]
    total_time = sum(times)
    average_time = total_time / len(times)
    print(
        f"  {experiment_id}: total={total_time:.2f}s, "
        f"average={average_time:.2f}s/epoch, epochs={len(times)}"
    )


def save_plot(filename):
    output_path = os.path.join(image_directory, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Wrote {output_path}")


def set_accuracy_limits(values):
    padding = 0.05
    plt.ylim(
        max(0.0, min(values) - padding),
        min(1.0, max(values) + padding),
    )


def line_style(experiment_id):
    color = "tab:blue" if experiment_id.startswith("conv") else "tab:orange"
    if ("batch_64" in experiment_id or "layers_5" in experiment_id
            or "lr_0001" in experiment_id):
        linestyle = "--"
    elif "sgd_momentum" in experiment_id:
        linestyle = ":"
    elif experiment_id.endswith("_sgd") or "_sgd" in experiment_id:
        linestyle = "--"
    elif "adamw" in experiment_id:
        linestyle = "-."
    else:
        linestyle = "-"
    linewidth = 2.5 if linestyle == "-" else 1.8
    marker = "o" if linestyle == "-" else "s"
    return color, linestyle, linewidth, marker


def plot_epoch_time(experiment_ids, filename, title, labels):
    
    selected_results = rows_for(experiment_ids)
    plt.figure(figsize=(8, 5))
    for experiment_id, label in zip(experiment_ids, labels):
        group = [
            row for row in selected_results
            if row["experiment_id"] == experiment_id
        ]
        times = [row["epoch_time_seconds"] for row in group]
        cumulative_times = []
        elapsed_time = 0.0
        for time_seconds in times:
            elapsed_time += time_seconds
            cumulative_times.append(elapsed_time)
        color, linestyle, linewidth, marker = line_style(experiment_id)
        plt.plot(
            [row["epoch"] for row in group],
            cumulative_times,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            label=label,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative training time (seconds)")
    plt.title(title)
    plt.legend()
    save_plot(filename)

# Learning rate comparison
selected = rows_for([
    "conv_baseline",
    "conv_lr_0001",
    "vit_baseline",
    "vit_lr_0001",
])
learning_rate_labels = {
    "conv_baseline": "CNN (LR=0.001)",
    "conv_lr_0001": "CNN (LR=0.0001)",
    "vit_baseline": "ViT (LR=0.001)",
    "vit_lr_0001": "ViT (LR=0.0001)",
}

for experiment_id in learning_rate_labels:
    group = [row for row in selected if row["experiment_id"] == experiment_id]
    color, linestyle, linewidth, marker = line_style(experiment_id)
    plt.plot(
        [row["epoch"] for row in group],
        [row["validation_accuracy"] for row in group],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        label=learning_rate_labels[experiment_id],
    )

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
set_accuracy_limits([row["validation_accuracy"] for row in selected])
plt.legend()
save_plot("learning_rate.png")


# Plain SGD versus Adam comparison
sgd_adam_ids = ["conv_sgd", "conv_baseline", "vit_sgd", "vit_baseline"]
sgd_adam_labels = {
    "conv_sgd": "CNN (SGD)",
    "conv_baseline": "CNN (Adam)",
    "vit_sgd": "ViT (SGD)",
    "vit_baseline": "ViT (Adam)",
}
sgd_adam_results = rows_for(sgd_adam_ids)
plt.figure(figsize=(8, 5))
for experiment_id in sgd_adam_ids:
    group = [
        row for row in sgd_adam_results
        if row["experiment_id"] == experiment_id
    ]
    color, linestyle, linewidth, marker = line_style(experiment_id)
    plt.plot(
        [row["epoch"] for row in group],
        [row["validation_accuracy"] for row in group],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        label=sgd_adam_labels[experiment_id],
    )
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Plain SGD versus Adam")
set_accuracy_limits([row["validation_accuracy"] for row in sgd_adam_results])
plt.legend()
save_plot("sgd_vs_adam.png")


def plot_final_accuracy(experiment_ids, filename, title, labels=None):
    comparison = [final_results[experiment_id] for experiment_id in experiment_ids]
    if labels is None:
        labels = experiment_ids
    plt.figure(figsize=(8, 5))
    plt.bar(
        labels,
        [row["validation_accuracy"] for row in comparison],
    )
    plt.title(title)
    plt.ylabel("Final validation accuracy")
    plt.xticks(rotation=30, ha="right")
    set_accuracy_limits([row["validation_accuracy"] for row in comparison])
    save_plot(filename)


plot_final_accuracy(
    ["linear_baseline", "three_layer_baseline", "conv_baseline", "vit_baseline"],
    "model_comparison.png",
    "Classifier comparison",
)

plot_final_accuracy(
    ["conv_baseline", "conv_batch_64", "vit_baseline", "vit_batch_64"],
    "batch_size.png",
    "Batch size comparison",
    labels=["CNN (128)", "CNN (64)", "ViT (128)", "ViT (64)"],
)
plot_epoch_time(
    ["conv_baseline", "conv_batch_64", "vit_baseline", "vit_batch_64"],
    "batch_time.png",
    "Batch size: time per epoch",
    ["CNN (128)", "CNN (64)", "ViT (128)", "ViT (64)"],
)

plot_final_accuracy(
    ["conv_sgd", "conv_sgd_momentum", "conv_baseline", "conv_adamw",
     "vit_sgd", "vit_sgd_momentum", "vit_baseline", "vit_adamw"],
    "optimizer_comparison.png",
    "Optimizer comparison",
    labels=[
        "CNN (SGD)", "CNN (SGD + momentum)", "CNN (Adam)", "CNN (AdamW)",
        "ViT (SGD)", "ViT (SGD + momentum)", "ViT (Adam)", "ViT (AdamW)",
    ],
)

plot_final_accuracy(
    ["conv_sgd", "conv_sgd_momentum", "conv_baseline", "conv_adamw"],
    "cnn_optimizer_comparison.png",
    "CNN optimizer comparison",
    labels=["SGD", "SGD + momentum", "Adam", "AdamW"],
)

plot_final_accuracy(
    ["vit_sgd", "vit_sgd_momentum", "vit_baseline", "vit_adamw"],
    "vit_optimizer_comparison.png",
    "ViT optimizer comparison",
    labels=["SGD", "SGD + momentum", "Adam", "AdamW"],
)
plot_epoch_time(
    ["conv_sgd", "conv_sgd_momentum", "conv_baseline", "conv_adamw"],
    "cnn_optimizer_time.png",
    "CNN optimizer: time per epoch",
    ["SGD", "SGD + momentum", "Adam", "AdamW"],
)
plot_epoch_time(
    ["vit_sgd", "vit_sgd_momentum", "vit_baseline", "vit_adamw"],
    "vit_optimizer_time.png",
    "ViT optimizer: time per epoch",
    ["SGD", "SGD + momentum", "Adam", "AdamW"],
)

depth_results = rows_for([
    "conv_baseline", "conv_layers_5", "vit_baseline", "vit_layers_5",
])
depth_ids = ["conv_baseline", "conv_layers_5", "vit_baseline", "vit_layers_5"]
depth_labels = {
    "conv_baseline": "CNN (2)",
    "conv_layers_5": "CNN (5)",
    "vit_baseline": "ViT (2)",
    "vit_layers_5": "ViT (5)",
}
plt.figure(figsize=(8, 4.5))
for experiment_id in depth_ids:
    group = [row for row in depth_results if row["experiment_id"] == experiment_id]
    color, linestyle, linewidth, marker = line_style(experiment_id)
    plt.plot(
        [row["epoch"] for row in group],
        [row["validation_accuracy"] for row in group],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        label=depth_labels[experiment_id],
    )
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Model depth comparison")
set_accuracy_limits([row["validation_accuracy"] for row in depth_results])
plt.legend()
save_plot("depth_comparison.png")
plot_epoch_time(
    ["conv_baseline", "conv_layers_5", "vit_baseline", "vit_layers_5"],
    "depth_time.png",
    "Model depth: time per epoch",
    ["CNN (2)", "CNN (5)", "ViT (2)", "ViT (5)"],
)