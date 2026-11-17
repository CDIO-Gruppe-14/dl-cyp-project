import json
from pathlib import Path
import matplotlib.pyplot as plt

# Finder alle Hydra run-mapper der har en history.jsonl
run_files = sorted(Path("outputs").rglob("history.jsonl"))

if not run_files:
    print("Fandt ingen history.jsonl under outputs/ – tjek mappestien!")
    exit()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for hist_path in run_files:
    run_dir = hist_path.parent
    cfg_path = run_dir / "config.json"

    # Aflæs weight_decay og modelnavn fra config.json hvis den findes
    label = str(run_dir.name)
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        wd = cfg.get("train", {}).get("weight_decay", "?")
        model_name = cfg.get("model", {}).get("name", cfg.get("model", {}).get("type", ""))
        label = f"{model_name} (wd={wd})"

    records = [json.loads(line) for line in hist_path.read_text().splitlines() if line.strip()]
    if not records:
        continue

    epochs = [r["epoch"] for r in records]
    train_loss = [r["train_loss"] for r in records]
    val_key = "macro_ST-RAE" if "macro_ST-RAE" in records[0] else "macro_MCC"
    val_score = [r[val_key] for r in records]

    ax1.plot(epochs, train_loss, marker="o", markersize=3, label=label)
    ax2.plot(epochs, val_score, marker="o", markersize=3, label=label)

    # Gem også en individuel graf inde i hver run-mappe
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, "r-o", markersize=3, label="Training loss")
    plt.plot(epochs, val_score, "b-o", markersize=3, label=f"Validation ({val_key})")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "training_curve.png", dpi=150)
    plt.close()

ax1.set_title("Training Loss vs. Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_title("Validation Score vs. Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Macro MCC / ST-RAE")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("weight_decay_comparison.png", dpi=150)
print(f"Gemte weight_decay_comparison.png og individuelle grafer for {len(run_files)} runs!")