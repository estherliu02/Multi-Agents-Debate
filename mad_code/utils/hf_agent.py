# code/utils/hf_agent.py
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# you can reuse or replace
from .openai_utils import num_tokens_from_string, model2max_context


class HFLocalAgent:
    """
    Drop-in replacement for the OpenAI Agent, but runs a local HF model instead.
    """

    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float = 0, device: str = None):
        self.model_name = model_name          # e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time

        # choose device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # load tokenizer + model once
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else None,
        )
        if device == "cpu":
            self.model.to(device)

        # if tokenizer has no pad token, set one
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ---------- memory handling (same idea as original Agent) ----------

    def set_meta_prompt(self, meta_prompt: str):
        self.memory_lst.append({"role": "system", "content": meta_prompt})

    def add_event(self, event: str):
        self.memory_lst.append({"role": "user", "content": event})

    def add_memory(self, memory: str):
        self.memory_lst.append({"role": "assistant", "content": memory})
        print(f"----- {self.name} -----\n{memory}\n")

    # ---------- chat -> prompt conversion ----------

    def _chat_to_prompt(self, messages):
        """
        Very simple chat template. For best results, adapt this to the model's
        recommended chat format (e.g. use tokenizer.apply_chat_template if available).
        """
        text = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                text += f"[SYSTEM] {content}\n"
            elif role == "user":
                text += f"[USER] {content}\n"
            elif role == "assistant":
                text += f"[ASSISTANT] {content}\n"
        # assume we want the assistant to speak next
        text += "[ASSISTANT] "
        return text

    # ---------- main generation method ----------

    def query(self, messages, max_tokens: int = 256, temperature: float = None) -> str:
        """
        Local generation with HF model. Similar signature to original Agent.query,
        but no api_key and no OpenAI errors.
        """
        time.sleep(self.sleep_time)
        if temperature is None:
            temperature = self.temperature

        prompt = self._chat_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # simple generation; tweak for your needs (top_p, top_k, etc.)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # cut off the prompt part
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_text = full_text[len(prompt):].strip()
        return gen_text

    def ask(self, temperature: float = None):
        """
        Same interface as original Agent.ask, but uses HF token counting.
        You can approximate tokens or re-use num_tokens_from_string if you want.
        """
        if temperature is None:
            temperature = self.temperature

        # approximate token count using HF tokenizer
        all_text = "".join(m["content"] for m in self.memory_lst)
        num_context_token = len(self.tokenizer.tokenize(all_text))

        # use model config if you don't have model2max_context
        max_context = getattr(
            self.model.config, "max_position_embeddings", 4096)
        max_new = max(16, max_context - num_context_token - 32)  # keep margin

        return self.query(self.memory_lst, max_tokens=max_new, temperature=temperature)
