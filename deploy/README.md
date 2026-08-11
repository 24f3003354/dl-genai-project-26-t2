---
title: Smart MCQ Solver
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Smart MCQ Solver

MLP trained on TF-IDF features using a pointwise binary classification approach.
Each question's 5 options are scored independently, then ranked to produce a Top-3 prediction.
