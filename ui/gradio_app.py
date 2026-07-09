"""
Gradio front end. Deliberately holds no model state itself -- every
button click makes an HTTP call to the FastAPI service (app/main.py),
which owns the ModelManager, generation, and scoring. This is what lets
the two run as separate processes/containers in docker-compose, or as
one container in the HF Space (see entrypoint.sh).
"""
import os

import gradio as gr
import matplotlib.pyplot as plt
import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api/v1")
TOPICS = ["World", "Sports", "Business", "Sci/Tech"]
DATA_SIZES = [25, 50, 100]
REQUEST_TIMEOUT = 120  # seconds; CPU decoding can be slow on a cold checkpoint load


def _raise_for_api_error(resp: requests.Response) -> None:
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise gr.Error(f"API error ({resp.status_code}): {detail}")


def generate_and_score(topic: str, size: int, n_samples: int):
    gen_resp = requests.post(
        f"{API_BASE}/generate",
        json={"topic": topic, "size": int(size), "n_samples": int(n_samples)},
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_api_error(gen_resp)
    texts = gen_resp.json()["texts"]

    eval_resp = requests.post(
        f"{API_BASE}/evaluate",
        json={"topic": topic, "texts": texts},
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_api_error(eval_resp)
    score = eval_resp.json()["style_sim"]

    formatted = "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return formatted, f"{score:.4f}"


def run_drift(topic: str, size: int):
    resp = requests.post(
        f"{API_BASE}/drift",
        json={"topic": topic, "size": int(size), "n_hops": 5},
        timeout=REQUEST_TIMEOUT,
    )
    _raise_for_api_error(resp)
    scores = resp.json()["scores"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(scores) + 1), scores, marker="o")
    ax.set_xlabel("Generation step (hop)")
    ax.set_ylabel("Style similarity")
    ax.set_title(f"{topic} \u2014 live 5-hop drift ({size}-line checkpoint)")
    ax.grid(True)
    return fig


def check_api_health() -> str:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        resident = ", ".join(data["resident_models"]) or "none yet"
        return f"API status: {data['status']} | resident checkpoints: {resident}"
    except requests.RequestException as exc:
        return f"API unreachable: {exc}"


with gr.Blocks(title="GPT-2 Style Emulation") as demo:
    gr.Markdown(
        "# GPT-2 Style Emulation\n"
        "This UI is a thin client over the FastAPI service -- generation, "
        "scoring, and model caching all happen server-side."
    )

    with gr.Tab("Generate & score"):
        with gr.Row():
            topic_in = gr.Dropdown(TOPICS, value="Sports", label="Topic / desk")
            size_in = gr.Dropdown(DATA_SIZES, value=100, label="Training size (n)")
            n_in = gr.Slider(1, 20, value=5, step=1, label="Samples to generate")
        run_btn = gr.Button("Generate", variant="primary")
        out_text = gr.Textbox(label="Generated headlines", lines=10)
        out_score = gr.Textbox(label="Mean StyleSim vs. held-out references")
        run_btn.click(generate_and_score, [topic_in, size_in, n_in], [out_text, out_score])

    with gr.Tab("Style drift (live, 5 hops)"):
        gr.Markdown(
            "Each hop feeds the model's own prior output back in as the next "
            "prompt (Eq. 4 in the paper). Watch register overlap rise, then decay."
        )
        with gr.Row():
            dtopic_in = gr.Dropdown(TOPICS, value="World", label="Topic / desk")
            dsize_in = gr.Dropdown(DATA_SIZES, value=100, label="Training size (n)")
        drift_btn = gr.Button("Run drift trace", variant="primary")
        drift_plot = gr.Plot()
        drift_btn.click(run_drift, [dtopic_in, dsize_in], drift_plot)

    with gr.Tab("API status"):
        status_box = gr.Textbox(label="Backend health", value=check_api_health)
        gr.Button("Refresh").click(check_api_health, outputs=status_box)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", 7860)))
