import gradio as gr

# Your existing action that starts the voice chat / kicks off inference
def start_voice_chat(mmsi: str, start_ts: str, end_ts: str):
    # TODO: call your existing logic here
    return f"Launching voice chat for MMSI {mmsi} | {start_ts} → {end_ts}"

with gr.Blocks() as demo:
    mmsi_in   = gr.Textbox(label="MMSI")
    start_in  = gr.Textbox(label="Start (YYYYMMDDTHHMMSS)")
    end_in    = gr.Textbox(label="End (YYYYMMDDTHHMMSS)")
    out       = gr.Markdown()
    go        = gr.Button("Start")

    go.click(start_voice_chat, inputs=[mmsi_in, start_in, end_in], outputs=[out])

    # Auto-prefill from URL and optionally auto-run
    def prefill_and_maybe_run(request: gr.Request):
        qp   = request.query_params or {}
        mmsi = qp.get("mmsi", "") or qp.get("MMSI", "")

        # Accept either start/end or startdate/enddate
        start = qp.get("start")
        end   = qp.get("end")
        sd    = qp.get("startdate")
        ed    = qp.get("enddate")

        if sd and ed:
            start_res = f"{sd}T000001"
            end_res   = f"{ed}T235959"
        else:
            start_res = start or ""
            end_res   = end or ""

        auto = (qp.get("auto", "1")).lower() in ("1", "true", "yes", "y")

        # Always prefill the inputs
        updates = [gr.update(value=mmsi), gr.update(value=start_res), gr.update(value=end_res)]

        # If all params exist and auto is true, run immediately
        if mmsi and start_res and end_res and auto:
            result = start_voice_chat(mmsi, start_res, end_res)
        else:
            result = gr.update()  # no change to output

        return (*updates, result)

    # On page load, prefill and (optionally) invoke
    demo.load(prefill_and_maybe_run, inputs=None, outputs=[mmsi_in, start_in, end_in, out], queue=False)

demo.launch()
