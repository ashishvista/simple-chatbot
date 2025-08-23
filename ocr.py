from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from PIL import Image

question = "Perform OCR  and convert it to markdown format."
image = Image.open("examples/bank-statement_page1.png")
model_name = "h2oai/h2ovl-mississippi-2b"


if __name__ == "__main__":
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)

    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

    # Stop tokens for H2OVL-Mississippi
    # https://huggingface.co/h2oai/h2ovl-mississippi-2b
    stop_token_ids = [tokenizer.eos_token_id]

    sampling_params = SamplingParams(n=1,
                                     temperature=0.8, 
                                     top_p=0.8,
                                     seed=777, # Seed for reprodicibility
                                     max_tokens=1024,
                                     stop_token_ids=stop_token_ids)

    # Single prompt inference
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    },
    sampling_params=sampling_params)

    # look at the output
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
