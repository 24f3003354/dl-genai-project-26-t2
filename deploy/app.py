import re
import spaces
import string
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

# ── Constants ─────────────────────────────────────────────────────────────────

OPTION_COLS  = ['A', 'B', 'C', 'D', 'E']
IDX_TO_OPT   = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

PREFIXES = [
    "choose the correct answer:",
    "determine the correct option:",
    "identify the correct statement:",
    "pick the best possible answer:",
    "select the most accurate option:",
    "which of the following is correct?",
]
SUFFIXES = [
    "from the following choices.",
    "carefully.",
    "based on the given context.",
    "among the listed options.",
]

# ── MLP architecture (must match training exactly) ────────────────────────────

class MCQ_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ── Load artifacts ────────────────────────────────────────────────────────────

INPUT_DIM = 2483  # fitted TF-IDF vocabulary size

tfidf = joblib.load("tfidf.pkl")

model = MCQ_MLP(input_dim=INPUT_DIM)
model.load_state_dict(torch.load("mlp.pt", map_location="cpu"))
model.eval()

print("Model and TF-IDF loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_prompt(text):
    text = str(text).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    lower = text.lower()
    for prefix in PREFIXES:
        if lower.startswith(prefix):
            text  = text[len(prefix):].strip()
            lower = text.lower()
            break
    for suffix in SUFFIXES:
        if lower.endswith(suffix):
            text  = text[:-len(suffix)].strip()
            lower = text.lower()
            break
    return re.sub(r'\s+', ' ', text)

def build_features(prompt, option_text):
    """Build TF-IDF features for one (prompt, option) pair."""
    return tfidf.transform([prompt + " " + option_text])

@spaces.GPU
def predict(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prompt   = clean_prompt(prompt)
    options  = [opt_a, opt_b, opt_c, opt_d, opt_e]

    scores = []
    for opt_text in options:
        X    = build_features(prompt, str(opt_text))
        X_t  = torch.tensor(X.toarray(), dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_t)
            prob   = F.softmax(logits, dim=-1)[0]
        # Score = probability of class index that corresponds to this option
        scores.append(float(prob.max()))   # or prob[1] if using pointwise binary head

    # Rank options by score
    ranked_idx  = np.argsort(scores)[::-1]
    ranked_opts = [OPTION_COLS[i] for i in ranked_idx]
    top3        = ranked_opts[:3]

    # Build output
    result_lines = []
    for rank, idx in enumerate(ranked_idx, 1):
        bar   = "█" * int(scores[idx] * 20)
        result_lines.append(
            f"{'⭐ ' if rank <= 3 else '   '}Rank {rank}  |  Option {OPTION_COLS[idx]}  |  {scores[idx]:.4f}  {bar}"
        )

    top3_str    = " ".join(top3)
    output_text = f"🏆  Top-3 Prediction:  {top3_str}\n\n" + "\n".join(result_lines)
    return output_text

# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Smart MCQ Solver", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧠 Smart MCQ Solver
    Enter a multiple-choice question and its five options.
    The model ranks all five options and returns the top-3 most likely correct answers.

    **Model:** MLP trained on TF-IDF features (pointwise binary classification)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(
                label    = "Question / Prompt",
                placeholder = "Enter the question here...",
                lines    = 3
            )
            with gr.Row():
                opt_a = gr.Textbox(label="Option A", placeholder="Option A text")
                opt_b = gr.Textbox(label="Option B", placeholder="Option B text")
            with gr.Row():
                opt_c = gr.Textbox(label="Option C", placeholder="Option C text")
                opt_d = gr.Textbox(label="Option D", placeholder="Option D text")
            with gr.Row():
                opt_e = gr.Textbox(label="Option E", placeholder="Option E text")

            submit_btn = gr.Button("🔍 Predict", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label  = "Prediction",
                lines  = 10,
                interactive = False
            )

    submit_btn.click(
        fn      = predict,
        inputs  = [prompt_box, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs = output_box
    )

    gr.Examples(
        examples = [[
            "Identify the correct statement: What is the effect generated by a spinning superconductor? among the listed options.",
            "An electric field, precisely aligned with the spin axis.",
            "A magnetic field, randomly aligned with the spin axis.",
            "A magnetic field, precisely aligned with the spin axis.",
            "A gravitational field, randomly aligned with the spin axis.",
            "A gravitational field, precisely aligned with the spin axis."
        ]],
        inputs = [prompt_box, opt_a, opt_b, opt_c, opt_d, opt_e]
    )

if __name__ == "__main__":
    demo.launch()
