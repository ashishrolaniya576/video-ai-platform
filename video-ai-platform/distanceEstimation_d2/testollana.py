# grocer_chat_gradio.py
import os, json, requests, gradio as gr
from typing import List, Dict, Generator

BASE_URL = os.environ.get(
    "OLLAMA_HOST",
    "https://cuisine-mainland-kitty-burn.trycloudflare.com/"
).rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "granite3.2:latest")

SYSTEM_PROMPT = """You are 'GroceryGuru', a helpful grocer assistant.
Goals:
- Help plan meals, suggest items, and build grocery lists.
- Ask clarifying questions (diet, budget, servings, cuisine, allergies).
- Provide concise, itemized outputs with quantities, units, and simple substitutions.
- When appropriate, group items by store section (Produce, Dairy, Pantry, Frozen, Snacks, Household).
- Be practical for India: consider common brands/sizes and local availability when relevant.
Format tips:
- Use short bullets.
- End with: 'Anything to add/remove?' when you propose a list."""

def _url(path: str) -> str:
    return f"{BASE_URL}{path}"

def chat_stream(messages: List[Dict[str, str]], temperature: float = 0.3) -> Generator[str, None, None]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }
    r = requests.post(_url("/api/chat"), json=payload, stream=True, timeout=300)
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
            delta = obj.get("message", {}).get("content", "")
            if delta:
                yield delta
            if obj.get("done"):
                break
        except json.JSONDecodeError:
            continue

def gradio_chat(user_msg, chat_history, temperature, budget, servings, allergies):
    # Build the rolling chat messages
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Inject soft context from UI controls (works like a priming assistant note)
    context_bits = []
    if budget: context_bits.append(f"Budget ≈ ₹{budget}")
    if servings: context_bits.append(f"Servings: {servings}")
    if allergies: context_bits.append(f"Allergies: {allergies}")
    if context_bits:
        msgs.append({"role":"assistant","content":"Context: " + "; ".join(context_bits)})

    # past turns
    for h in chat_history:
        msgs.append({"role":"user","content": h[0]})
        msgs.append({"role":"assistant","content": h[1]})
    # current turn
    msgs.append({"role":"user","content": user_msg})

    # Stream back to UI
    partial = ""
    for chunk in chat_stream(msgs, temperature=temperature):
        partial += chunk
        yield partial

with gr.Blocks(title="GroceryGuru (Ollama)") as demo:
    gr.Markdown("# 🛒 GroceryGuru — Grocer Assistant\nChatting via your Ollama endpoint.")

    with gr.Row():
        temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
        budget = gr.Number(label="Budget (₹)", value=None, precision=0)
        servings = gr.Number(label="Servings", value=None, precision=0)
        allergies = gr.Textbox(label="Allergies (comma-separated)", placeholder="peanuts, shellfish, gluten…")

    chat = gr.ChatInterface(
        fn=gradio_chat,
        additional_inputs=[temperature, budget, servings, allergies],
        examples=[
            ["Plan a 3-day vegetarian meal plan under ₹1500 with pantry staples."],
            ["Create a weekly grocery list for simple Indian breakfasts for 4 people."],
            ["Suggest dinner ideas for lactose-free and peanut-free diet, 3 nights."],
            ["I have tomatoes, onions, eggs, rice—what can I cook and what should I buy?"],
        ],
        title="GroceryGuru",
        multimodal=False
    )

if __name__ == "__main__":
    # Run on localhost:7860; change server_name to "0.0.0.0" if you need LAN access
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
