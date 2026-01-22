# ============================================
# AUTO-GENERATE PROJECT ASSETS (FIGURES)
# ============================================

import os
import matplotlib.pyplot as plt

# -------------------------------
# CREATE ASSETS DIRECTORY
# -------------------------------
ASSETS_DIR = "../assets/generated/figures"
os.makedirs(ASSETS_DIR, exist_ok=True)

# =====================================================
# MACHINE LEARNING PIPELINE DIAGRAM
# =====================================================
plt.figure(figsize=(14, 3))

pipeline_steps = [
    "Raw Data",
    "EDA",
    "Feature Engineering",
    "Model Training",
    "Evaluation",
    "Prediction"
]

for i, step in enumerate(pipeline_steps):
    plt.text(
        i, 0, step,
        ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.4")
    )

    if i < len(pipeline_steps) - 1:
        plt.arrow(i + 0.35, 0, 0.3, 0, head_width=0.05)

plt.axis("off")
plt.title("Machine Learning Pipeline Overview")

plt.savefig(
    f"{ASSETS_DIR}/ml_pipeline.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# =====================================================
# TRAINING vs INFERENCE WORKFLOW
# =====================================================
plt.figure(figsize=(12, 5))

# Training flow
plt.text(0, 1, "Historical Data", bbox=dict(boxstyle="round"))
plt.text(2, 1, "Feature Engineering", bbox=dict(boxstyle="round"))
plt.text(4, 1, "Train Model", bbox=dict(boxstyle="round"))
plt.text(6, 1, "Save Model", bbox=dict(boxstyle="round"))

# Inference flow
plt.text(2, 0, "New Incoming Data", bbox=dict(boxstyle="round"))
plt.text(4, 0, "Load Trained Model", bbox=dict(boxstyle="round"))
plt.text(6, 0, "Generate Prediction", bbox=dict(boxstyle="round"))

# Arrows (training)
plt.arrow(0.8, 1, 1, 0)
plt.arrow(2.8, 1, 1, 0)
plt.arrow(4.8, 1, 1, 0)

# Arrows (inference)
plt.arrow(2.8, 0, 1, 0)
plt.arrow(4.8, 0, 1, 0)

plt.axis("off")
plt.title("Training vs Inference Workflow")

plt.savefig(
    f"{ASSETS_DIR}/training_vs_inference.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# =====================================================
# MODEL ARCHITECTURE (LGBM vs XGBoost)
# =====================================================
plt.figure(figsize=(12, 4))

plt.text(1, 0.5, "Input Features\n(Time, Lag, Rolling Stats)",
         bbox=dict(boxstyle="round"))

plt.text(4, 0.8, "LightGBM\nGradient Boosting Trees",
         bbox=dict(boxstyle="round"))

plt.text(4, 0.2, "XGBoost\nGradient Boosting Trees",
         bbox=dict(boxstyle="round"))

plt.text(7, 0.5, "Predicted Energy Demand",
         bbox=dict(boxstyle="round"))

# Arrows
plt.arrow(2.3, 0.5, 1.2, 0.25)
plt.arrow(2.3, 0.5, 1.2, -0.25)
plt.arrow(5.9, 0.8, 1, -0.3)
plt.arrow(5.9, 0.2, 1, 0.3)

plt.axis("off")
plt.title("Model Architecture Overview")

plt.savefig(
    f"{ASSETS_DIR}/model_architecture.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("All asset figures successfully generated!")
