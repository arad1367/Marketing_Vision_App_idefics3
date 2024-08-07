import gradio as gr
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
import re
import time
from PIL import Image
import torch
import spaces
import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")

model = Idefics3ForConditionalGeneration.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True).to("cuda")

BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
EOS_WORDS_IDS = [processor.tokenizer.eos_token_id]

@spaces.GPU
def model_inference(
    images, text, assistant_prefix, decoding_strategy, temperature, max_new_tokens,
    repetition_penalty, top_p
):
    if text == "" and not images:
        gr.Error("Please input a query and optionally image(s).")

    if text == "" and images:
        gr.Error("Please input a text query along the image(s).")

    if isinstance(images, Image.Image):
        images = [images]

    resulting_messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}] + [
                        {"type": "text", "text": text}
                    ]
                }
            ]

    if assistant_prefix:
      text = f"{assistant_prefix} {text}"

    prompt = processor.apply_chat_template(resulting_messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[images], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    assert decoding_strategy in [
        "Greedy",
        "Top P Sampling",
    ]
    if decoding_strategy == "Greedy":
        generation_args["do_sample"] = False
    elif decoding_strategy == "Top P Sampling":
        generation_args["temperature"] = temperature
        generation_args["do_sample"] = True
        generation_args["top_p"] = top_p

    generation_args.update(inputs)

    # Generate
    generated_ids = model.generate(**generation_args)

    generated_texts = processor.batch_decode(generated_ids[:, generation_args["input_ids"].size(1):], skip_special_tokens=True)
    return generated_texts[0]

css = """
#app-container {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
#app-header {
    text-align: center;
    margin-bottom: 20px;
}
#app-description {
    text-align: center;
    margin-bottom: 20px;
}
#app-footer {
    text-align: center;
    margin-top: 20px;
    font-size: 0.9em;
}
#app-footer a {
    color: #007bff;
    text-decoration: none;
}
#app-footer a:hover {
    text-decoration: underline;
}
"""

with gr.Blocks(css=css, theme='JohnSmith9982/small_and_pretty') as demo:
    gr.HTML("<h1 id='app-header'>Marketing Vision App 📈</h1>")
    gr.HTML("<p id='app-description'>This app uses the <a href='https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3' target='_blank'>HuggingFaceM4/Idefics3-8B-Llama3</a> model to answer questions related to various topics. Upload an image and a text query.</p>")
    gr.Markdown("**Disclaimer:** This app may not consistently follow prompts or handle complex tasks. However, adding a prefix to the assistant's response can significantly improve the output. You could also play with the parameters such as the temperature in non-greedy mode.")
    with gr.Column():
        image_input = gr.Image(label="Upload your Image", type="pil", scale=1)
        query_input = gr.Textbox(label="Prompt")
        assistant_prefix = gr.Textbox(label="Assistant Prefix", placeholder="Let's think step by step.")

        submit_btn = gr.Button("Submit")
        output = gr.Textbox(label="Output")

    with gr.Accordion(label="Advanced Generation Parameters"):
        max_new_tokens = gr.Slider(
              minimum=8,
              maximum=1024,
              value=512,
              step=1,
              interactive=True,
              label="Maximum number of new tokens to generate",
          )
        repetition_penalty = gr.Slider(
              minimum=0.01,
              maximum=5.0,
              value=1.2,
              step=0.01,
              interactive=True,
              label="Repetition penalty",
              info="1.0 is equivalent to no penalty",
          )
        temperature = gr.Slider(
              minimum=0.0,
              maximum=5.0,
              value=0.4,
              step=0.1,
              interactive=True,
              label="Sampling temperature",
              info="Higher values will produce more diverse outputs.",
          )
        top_p = gr.Slider(
              minimum=0.01,
              maximum=0.99,
              value=0.8,
              step=0.01,
              interactive=True,
              label="Top P",
              info="Higher values is equivalent to sampling more low-probability tokens.",
          )
        decoding_strategy = gr.Radio(
              [
                  "Greedy",
                  "Top P Sampling",
              ],
              value="Greedy",
              label="Decoding strategy",
              interactive=True,
              info="Higher values is equivalent to sampling more low-probability tokens.",
          )
        decoding_strategy.change(
              fn=lambda selection: gr.Slider(
                  visible=(
                      selection in ["contrastive_sampling", "beam_sampling", "Top P Sampling", "sampling_top_k"]
                  )
              ),
              inputs=decoding_strategy,
              outputs=temperature,
          )

        decoding_strategy.change(
              fn=lambda selection: gr.Slider(
                  visible=(
                      selection in ["contrastive_sampling", "beam_sampling", "Top P Sampling", "sampling_top_k"]
                  )
              ),
              inputs=decoding_strategy,
              outputs=repetition_penalty,
          )
        decoding_strategy.change(
              fn=lambda selection: gr.Slider(visible=(selection in ["Top P Sampling"])),
              inputs=decoding_strategy,
              outputs=top_p,
          )

        submit_btn.click(model_inference, inputs = [image_input, query_input, assistant_prefix, decoding_strategy, temperature,
                                                      max_new_tokens, repetition_penalty, top_p], outputs=output)

    footer = """
    <div id='app-footer'>
        <a href='https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/' target='_blank'>LinkedIn</a> |
        <a href='https://github.com/arad1367' target='_blank'>GitHub</a> |
        <a href='https://arad1367.pythonanywhere.com/' target='_blank'>Live demo of my PhD defense</a>
        <br>
        Made with 💖 by Pejman Ebrahimi
    </div>
    """
    gr.HTML(footer)

demo.launch(debug=True)