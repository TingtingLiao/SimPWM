"""The VLM judge shared by the grasp and action-following evaluations and the Interact. metric: Qwen3-VL-8B-Instruct
(bf16, device_map auto), WorldArena's chat template (run_qwen_vl), greedy decoding.

The model is a Hugging Face id or a local snapshot (`--vlm` of the entry points, `use()`); jobs run offline, so it
must be in the HF cache."""

from __future__ import annotations

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_model = {"path": MODEL}


def use(path: str | None):
    """Select the model (None keeps the current one); it is loaded on first use."""
    if path and path != _model["path"]:
        _model.clear()
        _model["path"] = path


def name() -> str:
    return _model["path"]


def load():
    """(model, processor)."""
    if "vlm" not in _model:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path = _model["path"]
        model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        _model["vlm"] = (model, AutoProcessor.from_pretrained(path))
    return _model["vlm"]


def ask(content, max_new_tokens):
    """The answer to one user turn `content`: a list of {"type": "image", "image": PIL image} and
    {"type": "text", "text": ...} items."""
    import torch

    model, proc = load()
    inputs = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=proc.tokenizer.eos_token_id, pad_token_id=proc.tokenizer.eos_token_id)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
