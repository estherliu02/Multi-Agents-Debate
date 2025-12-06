# mad_code/utils/hf_agent.py

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFLocalAgent:
    """
    Drop-in replacement for the OpenAI Agent, but runs a local HF model instead.
    """

    def __init__(
        self,
        model_name: str,
        name: str,
        temperature: float,
        sleep_time: float = 0.0,
        device: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
    ):
        self.model_name = model_name          # e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time

        # choose device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ---------------- tokenizer selection ----------------
        # tokenizer_name can be different from model_name (for fine-tunes)
        if tokenizer_name is None:
            tokenizer_name = model_name

        # SPECIAL CASE: dishonest Llama fine-tune has broken tokenizer metadata.
        # Use the base Llama-3.1 tokenizer instead.
        if "hai_debate-dishonest_llama_3.1_8b_instruct" in model_name.lower():
            print("Using base Llama-3.1 tokenizer for dishonest model...")
            tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"

        # load tokenizer FIRST so we are guaranteed to have self.tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # ---------------- model loading ----------------
        # Use bfloat16 on GPU if available, otherwise default dtype on CPU
        model_kwargs = {}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["dtype"] = torch.bfloat16  # torch_dtype is deprecated
        else:
            model_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if device == "cpu":
            self.model.to(device)

        # ---------------- pad token handling ----------------
        # Many Llama / Gemma models have no pad token; use eos or add one.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # last resort: add a [PAD] token and resize embeddings
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))

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
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # cut off the prompt part
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_text = full_text[len(prompt):].strip()
        return gen_text

    def ask(self, temperature: float = None):
        """
        Same interface as original Agent.ask, but uses HF token counting.
        """
        if temperature is None:
            temperature = self.temperature

        # approximate token count using HF tokenizer
        all_text = "".join(m["content"] for m in self.memory_lst)
        num_context_token = len(self.tokenizer.tokenize(all_text))

        # use model config if available
        max_context = getattr(self.model.config, "max_position_embeddings", 4096)

        # HARD CAP for max_new_tokens (tune this!)
        hard_cap = 256  # or 128, 512, etc.

        max_new = max(16, max_context - num_context_token - 32)
        max_new = min(max_new, hard_cap)  # <- enforce cap

        return self.query(self.memory_lst, max_tokens=max_new, temperature=temperature)
