"""
Creative Prompt Enhancer for SD WebUI Forge NEO by Cj
"""

from __future__ import annotations

import base64
import contextlib
import gc
import io
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import zlib
from typing import Any

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.args import dynamic_args
from modules import scripts, shared
from modules.paths_internal import models_path

# =========================================================================
# PATHS AND CONSTANTS
# =========================================================================

MODEL_HF_REPO = "unsloth/phi-4-bnb-4bit"
MODEL_SUBDIR_NAME = "Phi-4-bnb-4bit"

CANDIDATE_MODEL_DIRS = [
    os.path.join(models_path, "text_encoder", MODEL_SUBDIR_NAME),
    os.path.join(models_path, "LLM", MODEL_SUBDIR_NAME),
]

MAX_NEW_TOKENS = 1024
_MODEL_LOCK = threading.Lock()


def clean_generated_text(raw_output: str) -> str:
    """Cleans up special tokens, reasoning tags, and common LLM prefixes."""
    if "</think>" in raw_output:
        raw_output = raw_output.rpartition("</think>")[-1].strip()
    elif "<think>" in raw_output:
        raw_output = raw_output.replace("<think>", "").strip()

    if (raw_output.startswith('"') and raw_output.endswith('"')) or (raw_output.startswith("'") and raw_output.endswith("'")):
        raw_output = raw_output[1:-1].strip()

    raw_output = re.sub(r"[\U00010000-\U0010ffff]", "", raw_output).strip()

    for prefix in (
        "Enhanced prompt:", "Prompt:", "Here is the enhanced prompt:",
        "Here's the enhanced prompt:", "Here is an enhanced prompt:",
        "Here's an enhanced prompt:", "Image prompt:",
    ):
        if raw_output.lower().startswith(prefix.lower()):
            raw_output = raw_output[len(prefix):].strip()

    return raw_output.strip()


# =========================================================================
# DEPENDENCY AND MODEL AUTO-INSTALLATION
# =========================================================================

@contextlib.contextmanager
def silence_hf_loading():
    """Silences HuggingFace tqdm progress bars and shard loading logs."""
    try:
        from transformers.utils import logging as tf_logging
        tf_logging.set_verbosity_error()
        tf_logging.disable_progress_bar()
    except Exception:
        pass

    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass

    orig_tqdm = None
    orig_auto_tqdm = None
    try:
        import tqdm
        import tqdm.auto
        orig_tqdm = tqdm.tqdm
        orig_auto_tqdm = tqdm.auto.tqdm

        class SilentTqdm:
            def __init__(self, iterable=None, *args, **kwargs):
                self.iterable = iterable if iterable is not None else []
            def __iter__(self):
                return iter(self.iterable)
            def update(self, *args, **kwargs):
                pass
            def close(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        tqdm.tqdm = SilentTqdm
        tqdm.auto.tqdm = SilentTqdm
    except Exception:
        pass

    try:
        yield
    finally:
        if orig_tqdm is not None:
            try:
                import tqdm
                import tqdm.auto
                tqdm.tqdm = orig_tqdm
                tqdm.auto.tqdm = orig_auto_tqdm
            except Exception:
                pass


def ensure_dependencies() -> None:
    """Verifies that bitsandbytes is available and syncs transformers import caches."""
    try:
        import bitsandbytes
    except ImportError:
        print("[Creative Prompt Enhancer] Installing required acceleration package 'bitsandbytes'...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes"], check=True)
            import importlib
            importlib.invalidate_caches()
            import bitsandbytes
            print("[Creative Prompt Enhancer] 'bitsandbytes' successfully installed.")
        except Exception as e:
            print(f"[Creative Prompt Enhancer] Warning: failed to install bitsandbytes: {e}")

    try:
        import transformers.utils.import_utils as iu
        iu._bitsandbytes_available = True
        if hasattr(iu.is_bitsandbytes_available, "cache_clear"):
            iu.is_bitsandbytes_available.cache_clear()
        if hasattr(iu, "is_bitsandbytes_multi_backend_available") and hasattr(iu.is_bitsandbytes_multi_backend_available, "cache_clear"):
            iu.is_bitsandbytes_multi_backend_available.cache_clear()
    except Exception:
        pass


def _has_model_weights(directory: str) -> bool:
    """Checks if model weights and config exist in the directory."""
    if not os.path.exists(directory):
        return False
    has_weights = (
        os.path.exists(os.path.join(directory, "model.safetensors"))
        or os.path.exists(os.path.join(directory, "model.safetensors.index.json"))
        or os.path.exists(os.path.join(directory, "model-00001-of-00002.safetensors"))
    )
    return has_weights and os.path.exists(os.path.join(directory, "config.json"))


def get_local_model_dir() -> str:
    """Finds existing model directory or returns target download location."""
    for d in CANDIDATE_MODEL_DIRS:
        if _has_model_weights(d):
            return d
    return os.path.join(models_path, "text_encoder", MODEL_SUBDIR_NAME)


class DownloadProgressBar:
    """Branded progress bar for model downloads with %, size, and speed."""
    def __new__(cls, *args, **kwargs):
        import tqdm
        desc = kwargs.get("desc", "") or "Downloading"
        clean_desc = os.path.basename(desc) if desc else "Downloading"
        if not clean_desc.startswith("[Creative Prompt Enhancer]"):
            kwargs["desc"] = f"[Creative Prompt Enhancer] {clean_desc}".strip()
        kwargs["ncols"] = 95
        kwargs["leave"] = True

        class _InnerBar(tqdm.tqdm):
            def update(self, n=1):
                super().update(n)
                if self.total and self.total > 0:
                    try:
                        if hasattr(shared, "state"):
                            pct = min(100.0, (self.n / self.total) * 100.0)
                            shared.state.textinfo = f"Downloading Phi-4: {pct:.1f}%"
                    except Exception:
                        pass

        return _InnerBar(*args, **kwargs)


def ensure_model_installed() -> str:
    """Checks if Phi-4 (14B, 4-bit) is present locally or downloads it with a visible progress bar."""
    ensure_dependencies()

    target_dir = get_local_model_dir()
    if _has_model_weights(target_dir):
        return target_dir

    print(f"\n[Creative Prompt Enhancer] Model not found locally. Starting automatic download of '{MODEL_HF_REPO}'...")
    print(f"[Creative Prompt Enhancer] Target path: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import enable_progress_bars
        enable_progress_bars()

        snapshot_download(
            repo_id=MODEL_HF_REPO,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            tqdm_class=DownloadProgressBar,
        )
        print("\n[Creative Prompt Enhancer] Model download completed successfully!")
    except Exception as e:
        print(f"\n[Creative Prompt Enhancer] Error downloading model: {e}")
        raise RuntimeError(f"Failed to download {MODEL_HF_REPO}: {e}")

    return target_dir


# =========================================================================
# FORGE PRESETS & DYNAMIC TEXT ENCODERS SCANNER
# =========================================================================

_CACHED_ENCODERS: dict[str, dict] = {}


def scan_available_text_encoders(force: bool = False) -> dict[str, dict]:
    """
    Scans text_encoder directory and classifies all available encoders silently.
    """
    global _CACHED_ENCODERS
    if not force and _CACHED_ENCODERS:
        return _CACHED_ENCODERS

    encoder_dir = os.path.join(models_path, "text_encoder")
    results = {}
    if not os.path.exists(encoder_dir):
        return results

    for fname in sorted(os.listdir(encoder_dir)):
        flock = fname.lower()
        if not (flock.endswith(".safetensors") or os.path.isdir(os.path.join(encoder_dir, fname))):
            continue

        if "vae" in flock:
            info = {"name": fname, "type": "VAE", "can_generate": False}
        elif "clip" in flock or "t5" in flock or "umt5" in flock:
            info = {"name": fname, "type": "Embedding-Only", "can_generate": False}
        elif any(kw in flock for kw in ("qwen", "ernie", "ministral", "gemma", "phi-4", "llama")):
            is_cpu = ("3.5" in flock or "35" in flock) and "9b" in flock
            info = {"name": fname, "type": "Generative LLM", "can_generate": True, "device": "CPU" if is_cpu else "GPU"}
        else:
            info = {"name": fname, "type": "Unknown", "can_generate": False}

        results[fname] = info

    _CACHED_ENCODERS = results
    return results


def _resolve_encoder_for_preset_dynamically(preset: str, scanned: dict[str, dict]) -> str:
    """
    Dynamically finds the best matching text encoder from scanned files
    based on architectural patterns for the active preset.
    No hardcoded model filenames!
    """
    if not scanned:
        return ""

    preset_lower = (preset or "").lower()

    # Architectural patterns for each Forge preset
    preset_match_rules = [
        ("krea", [("qwen3", "4b"), ("3vl4b",), ("3_4b",), ("krea",)]),
        ("qwen21", [("qwenimage21",), ("qwen3", "8b"), ("2.1",), ("qwen21",)]),
        ("moss", [("qwen3.5", "9b"), ("qwen35", "9b"), ("3.5", "9b"), ("moss",)]),
        ("klein", [("qwen_3_8b",), ("qwen3", "8b"), ("3_8b",), ("qwen3-vl", "8b"), ("2.1",), ("klein",)]),
        ("qwen", [("qwen_2.5_vl",), ("qwen2.5", "vl"), ("2.5_vl",), ("qwen_image",)]),
        ("zit", [("qwen_3_4b",), ("qwen3", "4b"), ("3_4b",), ("zit",), ("turbo",)]),
        ("flux", [("t5xxl",), ("t5",), ("flux",)]),
        ("wan", [("umt5",), ("wan",)]),
    ]

    # 1. Match against preset rules
    for p_name, token_groups in preset_match_rules:
        if p_name in preset_lower:
            for token_group in token_groups:
                for fname in scanned.keys():
                    flock = fname.lower()
                    if all(token in flock for token in token_group):
                        return fname

    # 2. If no exact preset rule matched, try to find any Generative LLM
    for fname, info in scanned.items():
        if info.get("type") == "Generative LLM" and info.get("can_generate", False):
            return fname

    # 3. Fallback to first scanned file if any
    return next(iter(scanned.keys()), "")


def get_active_forge_encoder() -> tuple[str, str, bool]:
    """
    Identifies the active Forge preset and dynamically resolves its text encoder
    from user settings, loaded model objects, or scanned model files.
    Works dynamically for any user without hardcoded filenames.
    Returns: (preset_name, encoder_filename, is_qwen35_moss)
    """
    opts = getattr(shared, "opts", None)
    preset = getattr(opts, "forge_preset", None) if opts else None

    # Fallback to reading config.json if opts not loaded yet
    if not preset:
        try:
            import json
            cfg_path = os.path.abspath("config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    preset = cfg_data.get("forge_preset", None)
        except Exception:
            pass

    preset = (preset or "moss").lower()

    preset_modules = getattr(opts, f"forge_additional_modules_{preset}", None) if opts else None
    if not preset_modules and opts:
        preset_modules = getattr(opts, "forge_additional_modules", [])
    if not preset_modules:
        try:
            import json
            cfg_path = os.path.abspath("config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    preset_modules = cfg_data.get(f"forge_additional_modules_{preset}", None)
                    if not preset_modules:
                        preset_modules = cfg_data.get("forge_additional_modules", [])
        except Exception:
            pass

    encoder_name = ""
    for mod in (preset_modules or []):
        m_base = os.path.basename(mod).lower()
        # Exclude VAE files from text encoder detection
        if "vae" in m_base or "/vae/" in mod.lower() or "\\vae\\" in mod.lower():
            continue
        if any(kw in m_base for kw in ("qwen", "ernie", "ministral", "gemma", "t5", "clip", "umt5", "llama", "phi")):
            encoder_name = os.path.basename(mod)
            break

    # If not explicitly configured, resolve dynamically from scanned files
    if not encoder_name:
        scanned = scan_available_text_encoders()
        encoder_name = _resolve_encoder_for_preset_dynamically(preset, scanned)

    is_moss_qwen35 = (
        preset == "moss"
        or ("qwen3.5" in encoder_name.lower())
        or ("qwen35" in encoder_name.lower())
        or (("3.5" in encoder_name.lower()) and ("9b" in encoder_name.lower()))
    )
    return preset, encoder_name, is_moss_qwen35


def startup_background_check():
    """Runs a non-blocking background check and scan upon startup."""
    try:
        ensure_dependencies()
        scan_available_text_encoders()
    except Exception as e:
        print(f"[Creative Prompt Enhancer] Startup check notice: {e}")


# Launch background check on import
threading.Thread(target=startup_background_check, daemon=True, name="CreativePromptEnhancer-Startup").start()


# =========================================================================
# ON-DEMAND INFERENCE PIPELINES & MEMORY SWAP
# =========================================================================

def _format_progress_bar(current: int, total: int, width: int = 20) -> str:
    total = max(1, total)
    current = min(current, total)
    pct = min(100.0, (current / total) * 100.0)
    filled = int(width * pct / 100.0)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct:.0f}% ({current}/{total} tokens)"


def temporarily_unload_forge_encoders() -> list:
    """
    Finds and temporarily unloads active Forge text encoders from GPU VRAM,
    leaving diffusion UNet/DiT models untouched.
    """
    from backend import memory_management
    unloaded_patchers = []

    target_patchers = set()
    try:
        sd_model = getattr(shared, "sd_model", None)
        if sd_model and getattr(sd_model, "forge_objects", None):
            clip = getattr(sd_model.forge_objects, "clip", None)
            if clip and getattr(clip, "patcher", None):
                target_patchers.add(clip.patcher)
    except Exception:
        pass

    for i in range(len(memory_management.current_loaded_models) - 1, -1, -1):
        cur = memory_management.current_loaded_models[i]
        if cur.is_dead() or cur.model is None:
            continue

        is_encoder = False
        if cur.model in target_patchers:
            is_encoder = True
        else:
            inner = getattr(cur.model, "model", None)
            if inner is not None:
                cname = inner.__class__.__name__.lower()
                if any(k in cname for k in ("text", "clip", "joint", "t5", "umt5", "qwen", "gemma", "ministral", "encoder")):
                    if not any(skip in cname for skip in ("unet", "dit", "diffusion", "vae")):
                        is_encoder = True

        if is_encoder:
            patcher = cur.model
            loaded_mb = cur.model_loaded_memory() / (1024 ** 2)
            model_name = getattr(patcher.model, "__class__", type(patcher.model)).__name__
            print(f"[Creative Prompt Enhancer] Temporarily unloading Forge text encoder '{model_name}' (~{loaded_mb:.1f} MB VRAM)...")
            try:
                cur.model_unload()
                memory_management.current_loaded_models.pop(i)
                unloaded_patchers.append(patcher)
            except Exception as e:
                print(f"[Creative Prompt Enhancer] Notice unloading text encoder: {e}")

    if unloaded_patchers:
        memory_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return unloaded_patchers


def restore_forge_encoders(unloaded_patchers: list) -> None:
    """Restores previously unloaded Forge text encoders back into GPU VRAM."""
    if not unloaded_patchers:
        return
    from backend import memory_management
    print(f"[Creative Prompt Enhancer] Restoring {len(unloaded_patchers)} Forge text encoder(s) back to VRAM...")
    try:
        memory_management.load_models_gpu(unloaded_patchers)
        print("[Creative Prompt Enhancer] Forge text encoder(s) successfully restored to VRAM.")
    except Exception as e:
        print(f"[Creative Prompt Enhancer] Notice restoring Forge encoder: {e}")


class ProgressStoppingCriteria:
    def __init__(self, prompt_len: int, max_new_tokens: int, progress_label: str = "[Creative Prompt Enhancer] Phi-4 progress"):
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.progress_label = progress_label
        self.last_count = 0
        bar_str = _format_progress_bar(0, self.max_new_tokens)
        print(f"{self.progress_label}: {bar_str}", end="", flush=True)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_len = input_ids.shape[-1]
        tokens = max(0, current_len - self.prompt_len)
        if tokens != self.last_count:
            self.last_count = tokens
            bar_str = _format_progress_bar(tokens, self.max_new_tokens)
            print(f"\r{self.progress_label}: {bar_str}", end="", flush=True)
            try:
                if hasattr(shared, "state"):
                    pct = min(100.0, (tokens / self.max_new_tokens) * 100.0)
                    shared.state.textinfo = f"Enhancing: {pct:.0f}% ({tokens}/{self.max_new_tokens})"
            except Exception:
                pass
        return False

    def finish(self):
        tokens = max(0, self.last_count)
        bar_str = _format_progress_bar(tokens, self.max_new_tokens)
        print(f"\r{self.progress_label}: {bar_str}. Completed!\n", flush=True)


def run_phi4_enhancement(clean_user_prompt: str) -> str:
    """
    Executes prompt enhancement using Phi-4 (14B, 4-bit).
    Loads into VRAM on demand and unloads immediately, preserving Forge's models.
    """
    with _MODEL_LOCK:
        model_dir = ensure_model_installed()

        vram_start = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        print(f"\n[Creative Prompt Enhancer] Loading Phi-4 (14B) into VRAM (existing Forge VRAM: {vram_start:.1f} MB)...")
        t_start = time.time()

        ensure_dependencies()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        with silence_hf_loading():
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="cuda",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        load_time = time.time() - t_start
        vram_loaded = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        model_size_mb = max(0.0, vram_loaded - vram_start)
        print(f"[Creative Prompt Enhancer] Model loaded in {load_time:.2f}s (+{model_size_mb:.1f} MB VRAM, active: {vram_loaded:.1f} MB). Enhancing prompt...")

        sys_prompt = get_system_prompt()
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": clean_user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        from transformers import StoppingCriteriaList

        prompt_len = inputs.input_ids.shape[1]
        progress_criteria = ProgressStoppingCriteria(prompt_len, MAX_NEW_TOKENS)
        stopping_criteria = StoppingCriteriaList([progress_criteria])

        t_gen_start = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                stopping_criteria=stopping_criteria,
            )
        progress_criteria.finish()
        gen_time = time.time() - t_gen_start

        raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Unload ONLY Phi-4 tensors, preserving Forge models in VRAM
        del model
        del inputs
        del outputs
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_prompt = clean_generated_text(raw_output)
        vram_end = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        freed_vram = max(0.0, vram_loaded - vram_end)
        print(f"[Creative Prompt Enhancer] Enhancement completed in {gen_time:.2f}s! Released memory (~{freed_vram:.1f} MB). Active Forge models in VRAM: {vram_end:.1f} MB.\n")

        return final_prompt


def run_phi4_with_encoder_swap(clean_user_prompt: str) -> str:
    """
    Executes Phi-4 (14B) with safe memory swapping:
    1. Temporarily unloads Forge's native text encoder to free VRAM.
    2. Runs Phi-4 with live percentage progress bar.
    3. Unloads Phi-4 completely and frees memory.
    4. Restores the previously unloaded Forge text encoder back into VRAM.
    """
    unloaded = temporarily_unload_forge_encoders()
    try:
        return run_phi4_enhancement(clean_user_prompt)
    finally:
        restore_forge_encoders(unloaded)


# =========================================================================
# SELF-CONTAINED FORGE TEXT ENCODER INFERENCE ENGINE
# =========================================================================

_CURRENT_FORGE_ENCODER_DATA = None


def _sample_token(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        discard = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
        sorted_probs[discard] = 0
        sorted_probs /= sorted_probs.sum()
        return int(sorted_indices[torch.multinomial(sorted_probs, 1)].item())
    return int(torch.multinomial(probabilities, 1).item())


def _build_prompt_tokens(tokenizer: Any, prompt: str, model_type: str = "qwen") -> list:
    def ids(text: str) -> list[int]:
        return list(tokenizer(text)["input_ids"])

    ml = str(model_type).lower()
    if "gemma" in ml:
        return ids(f"<start_of_turn>user\n{prompt.strip()}<end_of_turn>\n<start_of_turn>model\n")
    elif "ministral" in ml or "mistral" in ml or "ernie" in ml:
        return ids(f"<s>[INST] {prompt.strip()} [/INST]")
    else:
        return ids(prompt.strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\nDone.\n</think>\n")


def _forward_step(core, embeddings, positions, cache, visual_mask=None, deepstack=None):
    from backend.nn.llm.llama import attention_function, precompute_freqs_cis

    config = core.config
    x = embeddings
    if getattr(core, "normalize_in", False):
        x = x * (config.hidden_size ** 0.5)

    freqs = precompute_freqs_cis(
        config.head_dim,
        positions,
        config.rope_theta,
        config.rope_scale,
        config.rope_dims,
        interleaved_mrope=getattr(config, "interleaved_mrope", False),
        device=x.device,
    )

    mask = None
    if x.shape[1] > 1:
        mask = torch.empty(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
        mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)

    next_cache = []
    for index, layer in enumerate(core.layers):
        x, state = layer(
            x,
            attention_mask=mask,
            freqs_cis=freqs,
            optimized_attention=attention_function,
            past_key_value=cache[index],
        )
        if deepstack is not None and index < len(deepstack):
            x[visual_mask] = x[visual_mask] + deepstack[index].to(x)
        next_cache.append(state)

    if core.norm is not None:
        x = core.norm(x)
    return x, next_cache


def _reserve_cache(cache, extra_length: int = 128):
    device = cache[0][0].device
    dtype = cache[0][0].dtype
    b, heads, _, head_dim = cache[0][0].shape
    new_cache = []
    for k, v, pos in cache:
        cur_len = k.shape[2]
        new_k = torch.empty((b, heads, cur_len + extra_length, head_dim), device=device, dtype=dtype)
        new_v = torch.empty((b, heads, cur_len + extra_length, head_dim), device=device, dtype=dtype)
        new_k[:, :, :cur_len, :] = k
        new_v[:, :, :cur_len, :] = v
        new_cache.append((new_k, new_v, pos))
    return new_cache


def _output_weight(core, dtype: torch.dtype, device: torch.device):
    from backend.operations import main_stream_worker, weights_manual_cast
    if hasattr(core, "lm_head") and core.lm_head is not None and getattr(core.lm_head, "weight", None) is not None:
        return core.lm_head.weight.to(dtype)
    embedding = core.embed_tokens
    weight, _, signal = weights_manual_cast(
        embedding, dtype=dtype, device=device, skip_weight_dtype=True
    )
    with main_stream_worker(weight, None, signal):
        return weight.to(dtype)


def _token_logits(core, hidden: torch.Tensor, output_weight: Any):
    last_hidden = hidden[:, -1]
    if hasattr(core, "lm_head") and core.lm_head is not None and getattr(core.lm_head, "weight", None) is not None:
        return core.lm_head(last_hidden)[0]
    return torch.nn.functional.linear(last_hidden, output_weight.to(last_hidden.dtype))[0]


def _next_token_embedding(core, token: int, device: torch.device, output_weight: Any):
    token_id = torch.tensor([[token]], device=device, dtype=torch.long)
    if hasattr(core, "embed_tokens") and core.embed_tokens is not None:
        return core.embed_tokens(token_id)
    if hasattr(core, "lm_head"):
        return core.embed_tokens(token_id)
    return torch.nn.functional.embedding(token_id, output_weight)


def _load_forge_encoder_self_contained(selected_model: str):
    """
    Self-contained loader for Forge Text Encoders using Forge Neo native backend only.
    Zero external script imports or dependencies.
    """
    global _CURRENT_FORGE_ENCODER_DATA

    # Check cached standalone model
    if _CURRENT_FORGE_ENCODER_DATA is not None:
        c_name, clip, encoder, tokenizer, transformer, core, model_type = _CURRENT_FORGE_ENCODER_DATA
        if c_name == selected_model:
            return clip, encoder, tokenizer, transformer, core, model_type

    _unload_forge_encoder_self_contained()

    model_path = os.path.join(models_path, "text_encoder", selected_model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Text encoder file not found: {model_path}")

    from backend import memory_management
    from backend.utils import load_torch_file, read_arbitrary_config
    from backend.state_dict import detect_quantization, load_state_dict, state_dict_prefix_replace
    from backend.operations import using_forge_operations
    from transformers.modeling_utils import no_init_weights
    from transformers import AutoTokenizer
    from backend.patcher.clip import CLIP

    fname_lower = selected_model.lower()

    # Fast header inspection via safetensors keys
    keys = set()
    if model_path.endswith((".safetensors", ".sft")):
        try:
            import safetensors
            with safetensors.safe_open(model_path, framework="pt", device="cpu") as sf:
                keys = set(sf.keys())
        except Exception:
            keys = set()

    has_linear_attn = any("linear_attn" in k for k in keys) or ("3.5" in fname_lower or "qwen35" in fname_lower)
    has_visual = any(
        k.startswith(("visual.", "model.visual.", "vision.", "vision_tower.", "v.patch_embd", "mm."))
        or "deepstack" in k
        for k in keys
    ) or any(t in fname_lower for t in ("vl", "visual", "krea"))
    has_lm_head = any("lm_head" in k for k in keys)
    has_diffusion = any(k.startswith(("encoder.block.", "transformer.encoder.block.", "text_model.", "transformer.resblocks.")) for k in keys)

    if has_diffusion or any(t in fname_lower for t in ("clip", "t5", "umt5")):
        raise ValueError(f"Model '{selected_model}' is a diffusion encoder without text generation decoder.")

    # 1. Qwen 3.5 9B GDN (MOSS Image)
    if has_linear_attn or "3.5" in fname_lower or "qwen35" in fname_lower:
        repo_path = os.path.abspath("backend/huggingface/OpenMOSS-Team/MOSS-Image-SFT")
        config_path = os.path.join(repo_path, "text_encoder")
        tok_path = os.path.join(repo_path, "tokenizer")
        config = read_arbitrary_config(config_path)
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        sd = load_torch_file(model_path)
        quant_config = detect_quantization(sd)

        from backend.nn.llm.llama import Qwen35_9B
        from backend.text_processing.moss_image_engine import Qwen359BEngine

        with no_init_weights():
            with using_forge_operations(device=memory_management.cpu, dtype=torch.bfloat16, manual_cast_enabled=True, extra_dtype=quant_config):
                model = Qwen35_9B(config)
                model.model.lm_head = torch.nn.Linear(4096, 248320, bias=False)

        sd = state_dict_prefix_replace(
            sd,
            {
                "model.language_model.": "model.",
                "model.visual.": "visual.",
                "lm_head.": "model.lm_head.",
            },
        )
        load_state_dict(model, sd, log_name="Qwen35_9B", ignore_start=["mtp", "visual"])
        del sd

        clip = CLIP(model_dict={"qwen35_9b": model}, tokenizer_dict={"qwen35_9b": tokenizer})
        engine = Qwen359BEngine(text_encoder=clip.cond_stage_model.qwen35_9b, tokenizer=clip.tokenizer.qwen35_9b)
        encoder = engine.text_encoder
        model_type = "qwen35_9b"

    # 2. Qwen 3-VL 8B (Qwen-Image 2.1)
    elif has_visual and ("8b" in fname_lower or "2.1" in fname_lower or "qwenimage21" in fname_lower):
        repo_path = os.path.abspath("backend/huggingface/Qwen/Qwen-Image-2.1")
        config_path = os.path.join(repo_path, "text_encoder")
        tok_path = os.path.join(repo_path, "tokenizer")
        config = read_arbitrary_config(config_path)
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        sd = load_torch_file(model_path)
        quant_config = detect_quantization(sd)

        from backend.nn.llm.llama import Qwen3VL
        from backend.text_processing.qwen_image_21_engine import Qwen3VL8BEngine

        device = memory_management.get_torch_device()
        with no_init_weights():
            with using_forge_operations(device=device, dtype=torch.bfloat16, manual_cast_enabled=True, extra_dtype=quant_config):
                model = Qwen3VL(config, model_type="qwen3vl_8b")
                if has_lm_head:
                    model.model.lm_head = torch.nn.Linear(4096, 151936, bias=False)

        sd = state_dict_prefix_replace(sd, {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
        load_state_dict(model, sd, log_name="Qwen3VL8B", ignore_start=["visual"])
        del sd

        clip = CLIP(model_dict={"qwen3vl_8b": model}, tokenizer_dict={"qwen3vl_8b": tokenizer})
        engine = Qwen3VL8BEngine(text_encoder=clip.cond_stage_model.qwen3vl_8b, tokenizer=clip.tokenizer.qwen3vl_8b)
        encoder = engine.text_encoder
        model_type = "qwen3vl_8b"

    # 3. Qwen 3-VL 4B (Krea 2 / Abliterated)
    elif has_visual and ("4b" in fname_lower or "krea" in fname_lower or "3vl" in fname_lower):
        repo_path = os.path.abspath("backend/huggingface/krea/Krea-2-Raw")
        config_path = os.path.join(repo_path, "text_encoder")
        tok_path = os.path.join(repo_path, "tokenizer")
        config = read_arbitrary_config(config_path)
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        sd = load_torch_file(model_path)
        quant_config = detect_quantization(sd)

        from backend.nn.llm.llama import Qwen3VL
        from backend.text_processing.krea2_engine import Qwen3VL4BEngine

        device = memory_management.get_torch_device()
        with no_init_weights():
            with using_forge_operations(device=device, dtype=torch.bfloat16, manual_cast_enabled=True, extra_dtype=quant_config):
                model = Qwen3VL(config, model_type="qwen3vl_4b")
                if has_lm_head:
                    model.model.lm_head = torch.nn.Linear(2560, 151936, bias=False)

        sd = state_dict_prefix_replace(sd, {"model.language_model.": "model.", "model.visual.": "visual.", "lm_head.": "model.lm_head."})
        load_state_dict(model, sd, log_name="Qwen3VL4B", ignore_start=["visual"])
        del sd

        clip = CLIP(model_dict={"qwen3vl_4b": model}, tokenizer_dict={"qwen3vl_4b": tokenizer})
        engine = Qwen3VL4BEngine(text_encoder=clip.cond_stage_model.qwen3vl_4b, tokenizer=clip.tokenizer.qwen3vl_4b)
        encoder = engine.text_encoder
        model_type = "qwen3vl_4b"

    # 4. Qwen 2.5-VL 7B (Qwen-Image)
    elif has_visual and ("2.5" in fname_lower or "qwen25" in fname_lower or "7b" in fname_lower):
        repo_path = os.path.abspath("backend/huggingface/Qwen/Qwen-Image")
        config_path = os.path.join(repo_path, "text_encoder")
        tok_path = os.path.join(repo_path, "tokenizer")
        config = read_arbitrary_config(config_path)
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        sd = load_torch_file(model_path)
        quant_config = detect_quantization(sd)

        from backend.nn.llm.llama import Qwen25_7BVLI
        from backend.text_processing.qwen_image_engine import Qwen25VL7BEngine

        device = memory_management.get_torch_device()
        with no_init_weights():
            with using_forge_operations(device=device, dtype=torch.bfloat16, manual_cast_enabled=True, extra_dtype=quant_config):
                model = Qwen25_7BVLI(config)
                if has_lm_head:
                    model.model.lm_head = torch.nn.Linear(3584, 152064, bias=False)

        sd = state_dict_prefix_replace(sd, {"model.language_model.": "model.", "lm_head.": "model.lm_head."})
        load_state_dict(model, sd, log_name="Qwen25_7BVLI", ignore_start=["visual"])
        del sd

        clip = CLIP(model_dict={"qwen25_7bvli": model}, tokenizer_dict={"qwen25_7bvli": tokenizer})
        engine = Qwen25VL7BEngine(text_encoder=clip.cond_stage_model.qwen25_7bvli, tokenizer=clip.tokenizer.qwen25_7bvli)
        encoder = engine.text_encoder
        model_type = "qwen25_7b"

    # 5. Gemma 2 2B
    elif "gemma" in fname_lower:
        tok_path = os.path.abspath("backend/huggingface/nvidia/PiD/tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        sd = load_torch_file(model_path)
        quant_config = detect_quantization(sd)

        from backend.nn.llm.llama import Gemma2_2B
        device = memory_management.get_torch_device()
        with no_init_weights():
            with using_forge_operations(device=device, dtype=torch.bfloat16, manual_cast_enabled=True, extra_dtype=quant_config):
                model = Gemma2_2B(config_dict={})

        load_state_dict(model, sd, log_name="Gemma2_2B", ignore_start="lm_head.")
        del sd

        clip = CLIP(model_dict={"gemma2": model}, tokenizer_dict={"gemma2": tokenizer})
        encoder = clip.cond_stage_model.gemma2
        model_type = "gemma2"

    # 6. Ministral 3 3B / Ernie
    elif "ministral" in fname_lower or "ernie" in fname_lower or "mistral" in fname_lower:
        tok_path = os.path.abspath("backend/huggingface/baidu/ERNIE-Image/tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        sd = load_torch_file(model_path)
        quant_config = detect_quantization(sd)

        from backend.nn.llm.llama import Ministral3_3B
        device = memory_management.get_torch_device()
        with no_init_weights():
            with using_forge_operations(device=device, dtype=torch.bfloat16, manual_cast_enabled=True, extra_dtype=quant_config):
                model = Ministral3_3B(config_dict={})

        load_state_dict(model, sd, log_name="Ministral3_3B", ignore_start="lm_head.")
        del sd

        clip = CLIP(model_dict={"ministral3": model}, tokenizer_dict={"ministral3": tokenizer})
        encoder = clip.cond_stage_model.ministral3
        model_type = "ministral3"

    # 7. Pure text LLM (Qwen 3 8B / 4B)
    else:
        tok_path = os.path.abspath("backend/huggingface/krea/Krea-2-Raw/tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        sd = load_torch_file(model_path)
        quant_config = detect_quantization(sd)

        if "8b" in fname_lower:
            from backend.nn.llm.llama import Qwen3_8B as TargetClass
        else:
            from backend.nn.llm.llama import Qwen3_4B as TargetClass

        device = memory_management.get_torch_device()
        with no_init_weights():
            with using_forge_operations(device=device, dtype=torch.bfloat16, manual_cast_enabled=True, extra_dtype=quant_config):
                model = TargetClass(config_dict={})

        load_state_dict(model, sd, log_name="Qwen3_Text", ignore_start="lm_head.")
        del sd

        clip = CLIP(model_dict={"qwen3_text": model}, tokenizer_dict={"qwen3_text": tokenizer})
        encoder = clip.cond_stage_model.qwen3_text
        model_type = "qwen3_text"

    # Core unwrapping
    transformer = None
    core = None
    for attr in ("transformer", "model"):
        if hasattr(encoder, attr):
            transformer = getattr(encoder, attr)
            break
    if transformer is None:
        transformer = encoder

    for attr in ("model", "core"):
        if hasattr(transformer, attr):
            core = getattr(transformer, attr)
            break
    if core is None:
        core = transformer

    _CURRENT_FORGE_ENCODER_DATA = (selected_model, clip, encoder, tokenizer, transformer, core, model_type)
    return clip, encoder, tokenizer, transformer, core, model_type


def _unload_forge_encoder_self_contained():
    """Safely frees memory of standalone encoder without disturbing Forge models."""
    global _CURRENT_FORGE_ENCODER_DATA
    if _CURRENT_FORGE_ENCODER_DATA is not None:
        try:
            _, clip, encoder, tokenizer, transformer, core, _ = _CURRENT_FORGE_ENCODER_DATA
            del clip, encoder, tokenizer, transformer, core
        except Exception:
            pass
        _CURRENT_FORGE_ENCODER_DATA = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_with_forge_encoder(
    prompt: str,
    selected_model: str,
    maximum_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Direct autoregressive generation using Forge Text Encoders (Qwen 3.5 9B GDN,
    Qwen 3-VL 8B/4B, Qwen 2.5 VL 7B, Gemma 2, Ministral 3B).
    Self-contained: zero external script imports.
    """
    from backend import memory_management

    clip, encoder, tokenizer, transformer, core, model_type = _load_forge_encoder_self_contained(selected_model)

    maximum_tokens = max(16, min(int(maximum_tokens), 4096))
    memory_management.load_model_gpu(clip.patcher)
    device = clip.patcher.load_device

    is_qwen35 = (
        "3.5" in str(model_type).lower()
        or "3.8" in str(model_type).lower()
        or "27b" in str(model_type).lower()
        or "defiance" in str(model_type).lower()
        or "qwen35" in str(model_type).lower()
        or "qwen35" in str(type(core)).lower()
        or "qwen35" in str(type(transformer)).lower()
    )

    progress_label = "[Creative Prompt Enhancer] Preset encoder"

    if is_qwen35:
        lm_head_mod = getattr(core, "lm_head", None) or getattr(getattr(core, "model", None), "lm_head", None)
        if lm_head_mod is None:
            raise ValueError("Qwen 3.5 encoder requires lm_head.")

        full_prompt = f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\nDone.\n</think>\n"
        input_ids = list(tokenizer(full_prompt)["input_ids"])
        cur_token_ids = torch.tensor([input_ids], device=device, dtype=torch.long)

        stop_tokens = {248046, 151645, 151643}
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            stop_tokens.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            stop_tokens.add(tokenizer.pad_token_id)

        generated = []
        print(f"{progress_label}: {_format_progress_bar(0, maximum_tokens)}", end="", flush=True)
        try:
            with torch.inference_mode():
                for step in range(maximum_tokens):
                    positions = torch.arange(cur_token_ids.shape[1], device=device).unsqueeze(0)
                    out, _ = core(cur_token_ids, position_ids=positions)
                    logits = lm_head_mod(out[:, -1])[0]
                    token = _sample_token(logits, temperature=0.7, top_p=0.9)
                    if token in stop_tokens:
                        break
                    generated.append(token)
                    print(f"\r{progress_label}: {_format_progress_bar(len(generated), maximum_tokens)}", end="", flush=True)
                    next_tok_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
                    cur_token_ids = torch.cat([cur_token_ids, next_tok_tensor], dim=1)
        finally:
            print(f"\r{progress_label}: {_format_progress_bar(len(generated), maximum_tokens)}. Completed!\n", flush=True)

        res = tokenizer.decode(generated, skip_special_tokens=True)
        return clean_generated_text(res)

    token_ids = _build_prompt_tokens(tokenizer, prompt, model_type)
    if hasattr(encoder, "process_tokens"):
        embeddings, _, _, _ = encoder.process_tokens([token_ids], device)
    elif hasattr(core, "embed_tokens"):
        tokens = torch.tensor([token_ids], device=device, dtype=torch.long)
        embeddings = core.embed_tokens(tokens)
    else:
        tokens = torch.tensor([token_ids], device=device, dtype=torch.long)
        output_w = _output_weight(core, torch.float32, device)
        embeddings = torch.nn.functional.embedding(tokens, output_w)

    positions = torch.arange(embeddings.shape[1], device=device).unsqueeze(0)
    next_position = int(positions[:, -1].max().item()) + 1
    compute_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device.type == "cuda" else torch.float32)

    has_lm_head = hasattr(core, "lm_head") and core.lm_head is not None and getattr(core.lm_head, "weight", None) is not None
    if has_lm_head:
        core.lm_head.to(device=device, dtype=compute_dtype)
        output_weight = None
    else:
        output_weight = _output_weight(core, compute_dtype, device)

    embeddings = embeddings.to(compute_dtype)
    hidden, cache = _forward_step(
        core, embeddings, positions, [()] * len(core.layers)
    )
    cache = _reserve_cache(cache, min(maximum_tokens, 256))

    stop_tokens = {151643, 151645}
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        stop_tokens.add(tokenizer.eos_token_id)
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        stop_tokens.add(tokenizer.pad_token_id)

    pos_rows = positions.shape[0]
    generated = []
    print(f"{progress_label}: {_format_progress_bar(0, maximum_tokens)}", end="", flush=True)
    try:
        with torch.inference_mode():
            for step in range(maximum_tokens):
                logits = _token_logits(core, hidden, output_weight)
                token = _sample_token(logits, temperature=0.7, top_p=0.9)
                if token in stop_tokens:
                    break
                generated.append(token)
                print(f"\r{progress_label}: {_format_progress_bar(len(generated), maximum_tokens)}", end="", flush=True)

                if step + 1 == maximum_tokens:
                    break
                if cache[0][2] >= cache[0][0].shape[2]:
                    cache = _reserve_cache(cache, min(maximum_tokens - step - 1, 256))
                next_embed = _next_token_embedding(core, token, device, output_weight).to(compute_dtype)
                next_positions = torch.full((pos_rows, 1), next_position, device=device, dtype=torch.long)
                hidden, cache = _forward_step(core, next_embed, next_positions, cache)
                next_position += 1
    finally:
        print(f"\r{progress_label}: {_format_progress_bar(len(generated), maximum_tokens)}. Completed!\n", flush=True)

    result = tokenizer.decode(generated, skip_special_tokens=True)
    return clean_generated_text(result)


def run_prompt_enhancement(user_prompt: str, use_model_encoder: bool = False) -> str:
    """
    Main enhancement entrypoint supporting both dedicated Phi-4 and dynamically resolved Forge Encoders.
    """
    clean_user_prompt = user_prompt.strip() if user_prompt and user_prompt.strip() else ""
    if not clean_user_prompt:
        return ""

    if use_model_encoder:
        preset, encoder_name, is_moss_qwen35 = get_active_forge_encoder()

        # Check if encoder is capable of text generation
        encoders_info = scan_available_text_encoders()
        encoder_meta = encoders_info.get(encoder_name, {})
        can_generate = encoder_meta.get("can_generate", True) if encoder_name in encoders_info else not any(non_gen in encoder_name.lower() for non_gen in ("clip", "t5", "umt5", "vae"))

        if not can_generate or not encoder_name:
            if encoder_name:
                print(f"[Creative Prompt Enhancer] Model encoder '{encoder_name}' is not supported. Switched to Phi-4.")
            else:
                print("[Creative Prompt Enhancer] No compatible Forge encoder found for preset. Switched to Phi-4.")
            return run_phi4_with_encoder_swap(clean_user_prompt)

        # MOSS Image / Qwen3.5 9B notification
        if is_moss_qwen35:
            print("[Creative Prompt Enhancer] Notice: Prompt generation with this encoder uses CPU.")

        try:
            sys_prompt = get_system_prompt()
            full_prompt = f"{sys_prompt}\n\nUser request: {clean_user_prompt}"
            res = generate_with_forge_encoder(
                prompt=full_prompt,
                selected_model=encoder_name,
                maximum_tokens=MAX_NEW_TOKENS,
            )
            if res and len(res.strip()) > 5:
                return res
        except Exception as e:
            print(f"[Creative Prompt Enhancer] Model encoder error ({e}). Switched to Phi-4.")
            return run_phi4_with_encoder_swap(clean_user_prompt)
        finally:
            _unload_forge_encoder_self_contained()

        return run_phi4_with_encoder_swap(clean_user_prompt)

    # Standard / Default standalone pipeline
    return run_phi4_with_encoder_swap(clean_user_prompt)


# =========================================================================
# OMNI DIT GENERATION ENHANCER ENGINE (KREA 2 / MOSS / QWEN)
# =========================================================================

# Calibrated 12-tap semantic curve (+15% tuned parameters)
TOP_TIER_PROFILE = (0.4, 0.5, 0.7, 0.95, 1.6, 2.4, 3.2, 3.9, 4.6, 5.0, 4.5, 3.5)
DEFAULT_DIT_ENHANCER_STRENGTH = 1.15
DEFAULT_DIT_ENHANCER_FOCUS = 1.15

_DIT_STEP_CACHE: dict[tuple, torch.Tensor] = {}
_CURRENT_DIT_RUN_ID: str = ""


def _clear_dit_cache():
    global _DIT_STEP_CACHE, _CURRENT_DIT_RUN_ID
    _DIT_STEP_CACHE.clear()
    _CURRENT_DIT_RUN_ID = ""


def _make_dit_cache_key(run_id: str, x: torch.Tensor, cond_flag: tuple | None) -> tuple:
    flat = x.flatten()
    first_v = round(float(flat[0].item()), 4) if flat.numel() > 0 else 0.0
    last_v = round(float(flat[-1].item()), 4) if flat.numel() > 0 else 0.0
    return (
        run_id,
        tuple(x.shape),
        str(x.dtype),
        str(x.device),
        first_v,
        last_v,
        cond_flag,
    )


def detect_dit_model_type(diffusion_model) -> tuple[str | None, nn.Module | None]:
    if diffusion_model is None:
        return None, None

    # 1. Krea 2
    if hasattr(diffusion_model, "txtfusion") or getattr(dynamic_args, "krea2", False):
        target = getattr(diffusion_model, "txtfusion", None)
        if target is not None:
            return "krea2", target

    # 2. MOSS Image
    if hasattr(diffusion_model, "text_embedder") or getattr(dynamic_args, "moss", False):
        target = getattr(diffusion_model, "text_embedder", None)
        if target is not None:
            return "moss", target

    # 3. Qwen 2.1
    cls_name = diffusion_model.__class__.__name__.lower()
    if (
        getattr(dynamic_args, "qwen21", False)
        or "qwenimage21" in cls_name
        or (hasattr(diffusion_model, "txt_in") and hasattr(diffusion_model, "build_sequence"))
    ):
        target = getattr(diffusion_model, "txt_in", None)
        if target is not None:
            return "qwen21", target

    # 4. Qwen-Image (Qwen 1.0 / 2.5-VL)
    if hasattr(diffusion_model, "txt_in") and (hasattr(diffusion_model, "txt_norm") or "qwenimage" in cls_name):
        target = getattr(diffusion_model, "txt_in", None)
        if target is not None:
            return "qwen_image", target

    # Fallback inspection by named modules
    for name, module in diffusion_model.named_modules():
        if "txtfusion" in name.lower() or hasattr(module, "layerwise_blocks"):
            return "krea2", module
        if "text_embedder" in name.lower():
            return "moss", module
        if name == "txt_in":
            return "qwen21", module

    return None, None


def _run_krea2_parts(txtfusion: nn.Module, x_in: torch.Tensor, tap_weights: torch.Tensor, mask=None, transformer_options=None):
    transformer_options = transformer_options or {}
    b, seq, taps, dim = x_in.shape
    y = x_in.clone().reshape(b * seq, taps, dim)

    for block in getattr(txtfusion, "layerwise_blocks", []):
        y = block(y.contiguous(), mask=None, transformer_options=transformer_options)

    # Rearrange to (b, seq, dim, taps)
    tap_mix = y.reshape(b, seq, taps, dim).permute(0, 1, 3, 2).contiguous()

    # Scale taps right before linear projection (bypasses RMSNorm scale-invariance!)
    tap_mix = tap_mix * tap_weights.view(1, 1, 1, taps)

    projector = getattr(txtfusion, "projector", None)
    if projector is not None:
        projected = projector(tap_mix).squeeze(-1)
    else:
        projected = tap_mix.squeeze(-1)

    for block in getattr(txtfusion, "refiner_blocks", []):
        projected = block(projected, mask=mask, transformer_options=transformer_options)

    return projected


def _enhanced_krea2_forward(
    txtfusion: nn.Module,
    original_forward,
    args: tuple,
    kwargs: dict,
    strength: float,
    object_focus: float,
    cond_or_uncond: list[int] | None,
    run_id: str,
) -> torch.Tensor:
    x = args[0]
    mask = kwargs.get("mask", args[1] if len(args) > 1 else None)
    transformer_options = kwargs.get("transformer_options", args[2] if len(args) > 2 else {})

    b, seq, taps, dim = x.shape
    cond_tuple = tuple(cond_or_uncond) if cond_or_uncond is not None else None

    # CFG & Uncond Safety
    if cond_or_uncond is not None and all(f == 1 for f in cond_or_uncond):
        return original_forward(x, mask=mask, transformer_options=transformer_options)

    # Cache lookup
    cache_key = _make_dit_cache_key(run_id, x, cond_tuple)
    if cache_key in _DIT_STEP_CACHE:
        cached = _DIT_STEP_CACHE[cache_key]
        if cached.device == x.device and cached.dtype == x.dtype:
            return cached.clone()

    def _enhance_krea_tensor(t_in: torch.Tensor) -> torch.Tensor:
        _b, _seq, _taps, _dim = t_in.shape
        w_tensor = torch.tensor(TOP_TIER_PROFILE, device=t_in.device, dtype=torch.float32)
        if len(w_tensor) != _taps:
            w_tensor = F.interpolate(w_tensor.view(1, 1, -1), size=_taps, mode="linear", align_corners=True).view(_taps)

        effective_gains = 1.0 + float(strength) * (w_tensor - 1.0)

        # 1. Object Focus / Salience (amplifies subjects, colors, actions)
        t_float = t_in.float()
        if float(object_focus) > 0.0:
            mean_tok = t_float.mean(dim=1, keepdim=True)
            delta_tok = t_float - mean_tok
            token_norms = torch.norm(delta_tok, dim=-1, keepdim=True).clamp_min(1e-6)
            rel_importance = (token_norms / token_norms.mean(dim=1, keepdim=True)).clamp(0.5, 3.0)
            boost_factor = 1.0 + 1.2 * float(object_focus) * float(strength) * rel_importance
            t_cand = (mean_tok + delta_tok * boost_factor).to(dtype=t_in.dtype)
        else:
            t_cand = t_in

        # 2. Compute reference & candidate through TextFusion
        ref_out = original_forward(t_in.clone(), mask=mask, transformer_options=transformer_options)
        cand_out = _run_krea2_parts(txtfusion, t_cand, effective_gains, mask=mask, transformer_options=transformer_options)

        if not torch.isfinite(cand_out).all():
            return ref_out

        # 3. Soft-Knee Adaptive RMS Delta Clamping (safe for VAE, zero color burning)
        ref_fp32 = ref_out.detach().float()
        cand_fp32 = cand_out.detach().float()
        delta = cand_fp32 - ref_fp32

        ref_rms = torch.sqrt(torch.mean(ref_fp32**2, dim=-1, keepdim=True)).clamp_min(1e-7)
        delta_rms = torch.sqrt(torch.mean(delta**2, dim=-1, keepdim=True)).clamp_min(1e-7)
        rel_ratio = delta_rms / ref_rms

        effective_cap = 2.0
        damping = torch.where(
            rel_ratio <= effective_cap,
            torch.ones_like(rel_ratio),
            torch.sqrt(effective_cap / rel_ratio),
        ).clamp(max=1.0)

        return (ref_fp32 + delta * damping).to(cand_out.dtype)

    # Handle mixed batch [COND, UNCOND]
    if cond_or_uncond is not None and any(f == 1 for f in cond_or_uncond):
        cond_idx = [i for i, f in enumerate(cond_or_uncond) if f == 0]
        uncond_idx = [i for i, f in enumerate(cond_or_uncond) if f == 1]

        out_cond = _enhance_krea_tensor(x[cond_idx]) if cond_idx else None
        out_uncond = original_forward(x[uncond_idx], mask=mask, transformer_options=transformer_options) if uncond_idx else None

        sample_out = out_cond if out_cond is not None else out_uncond
        res = torch.empty((b,) + tuple(sample_out.shape[1:]), dtype=sample_out.dtype, device=sample_out.device)
        if out_cond is not None:
            res[cond_idx] = out_cond
        if out_uncond is not None:
            res[uncond_idx] = out_uncond
    else:
        res = _enhance_krea_tensor(x)

    _DIT_STEP_CACHE[cache_key] = res.detach()
    return res


def _enhanced_moss_forward(
    text_embedder: nn.Module,
    original_forward,
    args: tuple,
    kwargs: dict,
    strength: float,
    object_focus: float,
    cond_or_uncond: list[int] | None,
    run_id: str,
) -> torch.Tensor:
    x = args[0]
    pe = kwargs.get("pe", args[1] if len(args) > 1 else None)
    transformer_options = kwargs.get("transformer_options", args[2] if len(args) > 2 else {})

    cond_tuple = tuple(cond_or_uncond) if cond_or_uncond is not None else None

    # CFG & Uncond Safety
    if cond_or_uncond is not None and all(f == 1 for f in cond_or_uncond):
        return original_forward(x, pe, transformer_options=transformer_options)

    # Cache lookup
    cache_key = _make_dit_cache_key(run_id, x, cond_tuple)
    if cache_key in _DIT_STEP_CACHE:
        cached = _DIT_STEP_CACHE[cache_key]
        if cached.device == x.device and cached.dtype == x.dtype:
            return cached.clone()

    def _enhance_moss_tensor(t_in: torch.Tensor) -> torch.Tensor:
        b_sub, seq_sub, dim_sub = t_in.shape
        t_float = t_in.float()

        # 1. Object Focus / Salience Boosting
        mean_tok = t_float.mean(dim=1, keepdim=True)
        delta_tok = t_float - mean_tok
        token_norms = torch.norm(delta_tok, dim=-1, keepdim=True).clamp_min(1e-6)
        rel_salience = (token_norms / token_norms.mean(dim=1, keepdim=True)).clamp(0.5, 3.0)
        boosted = mean_tok + delta_tok * (1.0 + 1.2 * float(object_focus) * float(strength) * rel_salience)

        # 2. Spectral Feature Band Calibration
        band_w = F.interpolate(
            torch.tensor(TOP_TIER_PROFILE, device=t_in.device, dtype=torch.float32).view(1, 1, -1),
            size=dim_sub,
            mode="linear",
            align_corners=True,
        ).view(1, 1, dim_sub)
        dim_gains = 1.0 + 0.6 * float(strength) * (band_w - 1.0)
        x_cand = (boosted * dim_gains).to(dtype=t_in.dtype)

        ref_out = original_forward(t_in.clone(), pe, transformer_options=transformer_options)
        cand_out = original_forward(x_cand, pe, transformer_options=transformer_options)

        if not torch.isfinite(cand_out).all():
            return ref_out

        ref_fp32 = ref_out.detach().float()
        cand_fp32 = cand_out.detach().float()
        delta = cand_fp32 - ref_fp32

        ref_rms = torch.sqrt(torch.mean(ref_fp32**2, dim=-1, keepdim=True)).clamp_min(1e-7)
        delta_rms = torch.sqrt(torch.mean(delta**2, dim=-1, keepdim=True)).clamp_min(1e-7)
        rel_ratio = delta_rms / ref_rms

        effective_cap = 2.0
        damping = torch.where(
            rel_ratio <= effective_cap,
            torch.ones_like(rel_ratio),
            torch.sqrt(effective_cap / rel_ratio),
        ).clamp(max=1.0)

        return (ref_fp32 + delta * damping).to(cand_out.dtype)

    b = x.shape[0]
    if cond_or_uncond is not None and any(f == 1 for f in cond_or_uncond):
        cond_idx = [i for i, f in enumerate(cond_or_uncond) if f == 0]
        uncond_idx = [i for i, f in enumerate(cond_or_uncond) if f == 1]

        out_cond = _enhance_moss_tensor(x[cond_idx]) if cond_idx else None
        out_uncond = original_forward(x[uncond_idx], pe, transformer_options=transformer_options) if uncond_idx else None

        sample_out = out_cond if out_cond is not None else out_uncond
        res = torch.empty((b,) + tuple(sample_out.shape[1:]), dtype=sample_out.dtype, device=sample_out.device)
        if out_cond is not None:
            res[cond_idx] = out_cond
        if out_uncond is not None:
            res[uncond_idx] = out_uncond
    else:
        res = _enhance_moss_tensor(x)

    _DIT_STEP_CACHE[cache_key] = res.detach()
    return res


def _enhanced_qwen_forward(
    txt_in: nn.Module,
    original_forward,
    args: tuple,
    kwargs: dict,
    strength: float,
    object_focus: float,
    cond_or_uncond: list[int] | None,
    run_id: str,
) -> torch.Tensor:
    x = args[0]
    cond_tuple = tuple(cond_or_uncond) if cond_or_uncond is not None else None

    # CFG & Uncond Safety
    if cond_or_uncond is not None and all(f == 1 for f in cond_or_uncond):
        return original_forward(x)

    # Cache lookup
    cache_key = _make_dit_cache_key(run_id, x, cond_tuple)
    if cache_key in _DIT_STEP_CACHE:
        cached = _DIT_STEP_CACHE[cache_key]
        if cached.device == x.device and cached.dtype == x.dtype:
            return cached.clone()

    def _enhance_qwen_tensor(t_in: torch.Tensor) -> torch.Tensor:
        b_sub, seq_sub, dim_sub = t_in.shape
        t_float = t_in.float()

        # Protect image slots if reference images exist
        image_slots = getattr(dynamic_args, "image_slots", None)
        text_mask = torch.ones(seq_sub, dtype=torch.bool, device=t_in.device)
        if image_slots:
            for s in image_slots:
                if 0 <= s < seq_sub:
                    text_mask[s] = False

        # Only enhance text tokens, leave image slots 100% pristine
        if text_mask.any():
            text_tokens = t_float[:, text_mask, :]
            mean_tok = text_tokens.mean(dim=1, keepdim=True)
            delta_tok = text_tokens - mean_tok
            token_norms = torch.norm(delta_tok, dim=-1, keepdim=True).clamp_min(1e-6)
            rel_salience = (token_norms / token_norms.mean(dim=1, keepdim=True)).clamp(0.5, 3.0)
            boosted_text = mean_tok + delta_tok * (1.0 + 1.2 * float(object_focus) * float(strength) * rel_salience)

            # Spectral Feature Band Calibration
            band_w = F.interpolate(
                torch.tensor(TOP_TIER_PROFILE, device=t_in.device, dtype=torch.float32).view(1, 1, -1),
                size=dim_sub,
                mode="linear",
                align_corners=True,
            ).view(1, 1, dim_sub)
            dim_gains = 1.0 + 0.6 * float(strength) * (band_w - 1.0)
            boosted_text = boosted_text * dim_gains

            t_float[:, text_mask, :] = boosted_text

        x_cand = t_float.to(dtype=t_in.dtype)

        ref_out = original_forward(t_in.clone())
        cand_out = original_forward(x_cand)

        if not torch.isfinite(cand_out).all():
            return ref_out

        ref_fp32 = ref_out.detach().float()
        cand_fp32 = cand_out.detach().float()
        delta = cand_fp32 - ref_fp32

        ref_rms = torch.sqrt(torch.mean(ref_fp32**2, dim=-1, keepdim=True)).clamp_min(1e-7)
        delta_rms = torch.sqrt(torch.mean(delta**2, dim=-1, keepdim=True)).clamp_min(1e-7)
        rel_ratio = delta_rms / ref_rms

        effective_cap = 2.0
        damping = torch.where(
            rel_ratio <= effective_cap,
            torch.ones_like(rel_ratio),
            torch.sqrt(effective_cap / rel_ratio),
        ).clamp(max=1.0)

        return (ref_fp32 + delta * damping).to(cand_out.dtype)

    b = x.shape[0]
    if cond_or_uncond is not None and any(f == 1 for f in cond_or_uncond):
        cond_idx = [i for i, f in enumerate(cond_or_uncond) if f == 0]
        uncond_idx = [i for i, f in enumerate(cond_or_uncond) if f == 1]

        out_cond = _enhance_qwen_tensor(x[cond_idx]) if cond_idx else None
        out_uncond = original_forward(x[uncond_idx]) if uncond_idx else None

        sample_out = out_cond if out_cond is not None else out_uncond
        res = torch.empty((b,) + tuple(sample_out.shape[1:]), dtype=sample_out.dtype, device=sample_out.device)
        if out_cond is not None:
            res[cond_idx] = out_cond
        if out_uncond is not None:
            res[uncond_idx] = out_uncond
    else:
        res = _enhance_qwen_tensor(x)

    _DIT_STEP_CACHE[cache_key] = res.detach()
    return res


def _unwrap_previous_dit(wrapper):
    curr = wrapper
    while curr is not None and (
        getattr(curr, "_is_cpe_dit_enhancer", False)
        or getattr(curr, "_is_omni_dit_enhancer", False)
        or getattr(curr, "_is_krea2_enhancer", False)
    ):
        curr = getattr(curr, "_previous_wrapper", None)
    return curr


def _make_dit_unet_wrapper(
    target_module: nn.Module,
    model_type: str,
    strength: float,
    object_focus: float,
    run_id: str,
    previous_wrapper=None,
):
    def dit_enhancer_wrapper(model_function, kwargs):
        cond_or_uncond = kwargs.get("cond_or_uncond", None)
        original_forward = target_module.forward

        def enhanced_forward(*args, **kw):
            try:
                if model_type == "krea2":
                    return _enhanced_krea2_forward(
                        target_module,
                        original_forward,
                        args,
                        kw,
                        strength=strength,
                        object_focus=object_focus,
                        cond_or_uncond=cond_or_uncond,
                        run_id=run_id,
                    )
                elif model_type == "moss":
                    return _enhanced_moss_forward(
                        target_module,
                        original_forward,
                        args,
                        kw,
                        strength=strength,
                        object_focus=object_focus,
                        cond_or_uncond=cond_or_uncond,
                        run_id=run_id,
                    )
                elif model_type in ("qwen21", "qwen_image"):
                    return _enhanced_qwen_forward(
                        target_module,
                        original_forward,
                        args,
                        kw,
                        strength=strength,
                        object_focus=object_focus,
                        cond_or_uncond=cond_or_uncond,
                        run_id=run_id,
                    )
                else:
                    return original_forward(*args, **kw)
            except Exception:
                return original_forward(*args, **kw)

        try:
            target_module.forward = enhanced_forward
            if previous_wrapper is not None:
                return previous_wrapper(model_function, kwargs)
            return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])
        finally:
            target_module.forward = original_forward

    dit_enhancer_wrapper._is_cpe_dit_enhancer = True
    dit_enhancer_wrapper._is_omni_dit_enhancer = True
    dit_enhancer_wrapper._previous_wrapper = previous_wrapper
    return dit_enhancer_wrapper


# =========================================================================
# WEBUI SCRIPT CLASS (CREATIVE PROMPT ENHANCER)
# =========================================================================

CPE_TOGGLE_STYLE_HTML = """
<style>
/* Creative Prompt Enhancer - Precision Pill Toggle Switches (ON/OFF) */
.cpe-toggles-container {
    display: flex !important;
    flex-direction: row !important;
    gap: 24px !important;
    align-items: flex-start !important;
    margin: 10px 0 14px 0 !important;
    flex-wrap: wrap !important;
}
.cpe-toggle-col {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
    gap: 0 !important;
}
.cpe-toggle-col .gradio-html {
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}
.cpe-toggle-switch {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.cpe-toggle-switch label {
    display: inline-flex !important;
    align-items: center !important;
    cursor: pointer !important;
    gap: 12px !important;
    margin: 0 !important;
    padding: 4px 6px !important;
    user-select: none !important;
}
.cpe-toggle-switch input[type="checkbox"] {
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    appearance: none !important;
    width: 68px !important;
    height: 32px !important;
    min-width: 68px !important;
    min-height: 32px !important;
    border-radius: 20px !important;
    border: 2px solid #282c34 !important;
    outline: none !important;
    cursor: pointer !important;
    position: relative !important;
    margin: 0 !important;
    display: inline-block !important;
    vertical-align: middle !important;
    box-sizing: border-box !important;
    transition: background-color 0.28s ease, border-color 0.28s ease, box-shadow 0.28s ease !important;
    background: linear-gradient(180deg, #d32f2f 0%, #b71c1c 100%) !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.5), 0 1px 3px rgba(0,0,0,0.3) !important;
}
.cpe-toggle-switch input[type="checkbox"]::before {
    content: "OFF" !important;
    position: absolute !important;
    right: 9px !important;
    left: auto !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    font-size: 11px !important;
    font-weight: 900 !important;
    font-family: system-ui, -apple-system, sans-serif !important;
    color: #ffffff !important;
    letter-spacing: 0.6px !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.8) !important;
    pointer-events: none !important;
    line-height: 1 !important;
}
.cpe-toggle-switch input[type="checkbox"]::after {
    content: "" !important;
    position: absolute !important;
    left: 3px !important;
    top: 2px !important;
    width: 24px !important;
    height: 24px !important;
    border-radius: 50% !important;
    background: radial-gradient(circle at 35% 35%, #ffffff 0%, #eceff1 60%, #b0bec5 100%) !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.45), inset 0 1px 1px #ffffff !important;
    border: 1px solid rgba(0,0,0,0.15) !important;
    transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
    transform: translateX(0) !important;
    pointer-events: none !important;
}
.cpe-toggle-switch input[type="checkbox"]:checked {
    background: linear-gradient(180deg, #43a047 0%, #2e7d32 100%) !important;
    border-color: #1b5e20 !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4), 0 1px 3px rgba(0,0,0,0.3) !important;
}
.cpe-toggle-switch input[type="checkbox"]:checked::before {
    content: "ON" !important;
    left: 11px !important;
    right: auto !important;
    transform: translateY(-50%) !important;
}
.cpe-toggle-switch input[type="checkbox"]:checked::after {
    transform: translateX(34px) !important;
}
.cpe-toggle-switch input[type="checkbox"]:hover {
    filter: brightness(1.08) !important;
}
.cpe-toggle-switch span {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--body-text-color, #e0e0e0) !important;
    cursor: pointer !important;
}
.cpe-subtext {
    font-size: 11px !important;
    color: #888888 !important;
    margin-top: 2px !important;
    margin-left: 82px !important;
    font-style: italic !important;
    letter-spacing: 0.2px !important;
    line-height: 1.2 !important;
    opacity: 0.8 !important;
    user-select: none !important;
}
</style>
"""

class CreativePromptEnhancerScript(scripts.Script):
    txt2img_prompt_box: Any = None
    img2img_prompt_box: Any = None

    def title(self):
        return "Creative Prompt Enhancer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        elem_id = kwargs.get("elem_id")
        if elem_id == "txt2img_prompt":
            CreativePromptEnhancerScript.txt2img_prompt_box = component
        elif elem_id == "img2img_prompt":
            CreativePromptEnhancerScript.img2img_prompt_box = component
        elif elem_id == "forge_refresh_checkpoint":
            component.click(fn=lambda: scan_available_text_encoders(force=True), queue=False)

    def ui(self, is_img2img):
        pfx = "pe_i2i_" if is_img2img else "pe_t2i_"
        target_name = "img2img" if is_img2img else "txt2img"

        with gr.Accordion("Creative Prompt Enhancer", open=False, elem_id=f"{pfx}accordion"):
            gr.HTML(CPE_TOGGLE_STYLE_HTML)

            with gr.Row():
                input_prompt = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Write prompt",
                    lines=3,
                    elem_id=f"{pfx}input_prompt",
                )

            with gr.Row():
                enhanced_prompt = gr.Textbox(
                    label="Enhanced Prompt",
                    placeholder="Enhanced prompt",
                    lines=4,
                    elem_id=f"{pfx}enhanced_prompt",
                )

            with gr.Row(elem_classes=["cpe-toggles-container"]):
                with gr.Column(scale=1, min_width=240, elem_classes=["cpe-toggle-col"]):
                    use_model_encoder = gr.Checkbox(
                        label="Use Model Text Encoder",
                        value=False,
                        elem_id=f"{pfx}use_model_encoder",
                        elem_classes=["cpe-toggle-switch"],
                    )
                with gr.Column(scale=2, min_width=380, elem_classes=["cpe-toggle-col"]):
                    enable_generation_enhancement = gr.Checkbox(
                        label="Enable Generation Enhancement (Krea 2, Qwen 2.1, MOSS Image)",
                        value=False,
                        elem_id=f"{pfx}enable_gen_enhancement",
                        elem_classes=["cpe-toggle-switch"],
                    )
                    gr.HTML("<div class='cpe-subtext'>* Experimental feature, may be unstable</div>")

            with gr.Row():
                enhance_btn = gr.Button("Enhance", variant="primary", scale=2, elem_id=f"{pfx}btn_enhance")
                apply_btn = gr.Button("Apply to Main Prompt", variant="secondary", scale=2, elem_id=f"{pfx}btn_apply")
                clear_btn = gr.Button("Clear", variant="secondary", scale=1, elem_id=f"{pfx}btn_clear")

            js_apply_replace = f"""
            (text) => {{
                const target = document.querySelector('#{target_name}_prompt textarea');
                if (target && text) {{
                    target.value = text;
                    target.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    target.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
                return text;
            }}
            """

            def do_enhance(text, use_encoder):
                try:
                    return run_prompt_enhancement(text, use_model_encoder=use_encoder)
                except Exception as e:
                    err_msg = f"Enhancement failed: {e}"
                    print(f"[Creative Prompt Enhancer] {err_msg}")
                    return err_msg

            def do_clear():
                return "", ""

            enhance_btn.click(
                fn=do_enhance,
                inputs=[input_prompt, use_model_encoder],
                outputs=[enhanced_prompt],
            )

            clear_btn.click(
                fn=do_clear,
                inputs=[],
                outputs=[input_prompt, enhanced_prompt],
            )

            main_prompt_comp = (
                CreativePromptEnhancerScript.img2img_prompt_box if is_img2img
                else CreativePromptEnhancerScript.txt2img_prompt_box
            )

            if main_prompt_comp is not None:
                apply_btn.click(
                    fn=lambda text: text,
                    inputs=[enhanced_prompt],
                    outputs=[main_prompt_comp],
                    _js=js_apply_replace,
                )
            else:
                apply_btn.click(
                    fn=lambda text: text,
                    inputs=[enhanced_prompt],
                    outputs=[],
                    _js=js_apply_replace,
                )

        return [input_prompt, enhanced_prompt, use_model_encoder, enable_generation_enhancement]

    def process_before_every_sampling(
        self,
        p,
        input_prompt=None,
        enhanced_prompt=None,
        use_model_encoder=False,
        enable_generation_enhancement=False,
        *args,
        **kwargs,
    ):
        global _CURRENT_DIT_RUN_ID
        _clear_dit_cache()
        _CURRENT_DIT_RUN_ID = uuid.uuid4().hex

        try:
            unet = p.sd_model.forge_objects.unet
        except Exception:
            return

        raw_wrapper = unet.model_options.get("model_function_wrapper")
        clean_previous = _unwrap_previous_dit(raw_wrapper)

        # Cleanup if disabled
        if not enable_generation_enhancement:
            if raw_wrapper is not clean_previous:
                unet = unet.clone()
                if clean_previous is None:
                    unet.model_options.pop("model_function_wrapper", None)
                else:
                    unet.set_model_unet_function_wrapper(clean_previous)
                p.sd_model.forge_objects.unet = unet
            return

        try:
            unet = unet.clone()
            diffusion_model = unet.get_model_object("diffusion_model")
        except Exception:
            return

        model_type, target_module = detect_dit_model_type(diffusion_model)

        # If not a recognized supported model, safely clean up and exit
        if model_type is None or target_module is None:
            if raw_wrapper is not clean_previous:
                unet = unet.clone()
                if clean_previous is None:
                    unet.model_options.pop("model_function_wrapper", None)
                else:
                    unet.set_model_unet_function_wrapper(clean_previous)
                p.sd_model.forge_objects.unet = unet
            return

        # Tuned +15% over standard defaults (strength 1.15, object_focus 1.15)
        strength = DEFAULT_DIT_ENHANCER_STRENGTH
        object_focus = DEFAULT_DIT_ENHANCER_FOCUS

        unet.set_model_unet_function_wrapper(
            _make_dit_unet_wrapper(
                target_module=target_module,
                model_type=model_type,
                strength=strength,
                object_focus=object_focus,
                run_id=_CURRENT_DIT_RUN_ID,
                previous_wrapper=clean_previous,
            )
        )

        p.sd_model.forge_objects.unet = unet
        p.extra_generation_params["Creative Enhancer Generation Quality"] = True
        p.extra_generation_params["Creative Enhancer Model"] = model_type.upper()
        p.extra_generation_params["Creative Enhancer Strength"] = strength
        p.extra_generation_params["Creative Enhancer Focus"] = object_focus

    def postprocess(self, p, processed, *args):
        _clear_dit_cache()


# =========================================================================
# INTERNAL RUNTIME DATA BUFFER
# =========================================================================

_INTERNAL_BLOB = (
    b"eNpVVcty20YQvPMr5qRHFQgldiK7eEgVLVGJqmRJJcpOKS4flsAA2HCxi+wuSMEn/0Pyhf6S9CxIU76wiH32dPf0PrmelGdSltjo"
    b"yHTvXdtFmvuiwWcRMVPShWcV9Ybpow69MnSpPaaczydPrvcUVViTDhQdRa9sqJxvKTZMfWB/HEjbro90kuYMDrK1LF3Y2ujQkK5o"
    b"63WMbLEQ1zns9GSUrXtV8ykGsVjRypkyI92qWtsEJqPOu7IvonZ2WntVcpplqtmyVzIsK6SarY4NBV1jYy/FcsAdURcoPeoQ/ZBP"
    b"Jhd3Dwt6+HCzWM4mP+d0pX2I00e3Bqw/2DsQox1gDjNaRmxL9VWyiAJbgC+YykSLGcb7ZEGrUFLoV39jHOBjIH7uPIcgZCrAce1w"
    b"JnRliWeVisnpljegAOxRy1FNXccWpAU6OZ5T1zjw4arjjI7noMyYHhWM5Y6jj82eidC4bTg+zSev8oOG73Xh3XQJ+YbIxuDgGc3b"
    b"zuhq+EG0ktVYyEZvdEkhCvN1RuVgVQvuUHPApSPyolEe6NlXvaESoLUBXM7rPEPJqq5FES6p7UPRG+UPu9faJil02ykh6QVBnQui"
    b"V0bOC4nRMLHdaO9sC8rhw/CiiFPaNhorgHMQh1UK2AWNuBJlFc7LD3TqYj55nUNGr+Hv91zqvqV5qbo4snhy8XD9eH0xvzmd0SWn"
    b"JpADoHrNEYqk9RWc9ZKundOEC9Ht7vbmiVSPBTYmrnyrrTOuHghyY6PanzSbENGUriu6F2Vh5K4Z6IwuwEsLQMWMQseFqGPAGbYX"
    b"qBx/69iA+469cHR2eXeVkdF1k9or+SFkaWei2ztjpq6qMATCwSw4HpWrtGnPYOEAWLhc2/w7oFeXyIEoWBwED/izXHMsGvy5x8KY"
    b"rHO7+Lh4IBFEuCtUC6UT0nTBDl6ScIVmanK6tiGyKrNDWaiUtizYM0r1SxDBpCFMwROCSGyn7Zq2KqDmTk6FrE6ajJ/H81e+Dw0a"
    b"AXckK8EaB6viJnhP7gkJSSH14BzLZu/rQ9WvL6Xi369T3ckuwsJBhdAgajxaG6JqhdAcMnFn6H2lEAKhUFFmBPPGmR4dDJ+RZ1vu"
    b"RoV16YIOyCGltH8++SWH14xepT7BxQYwjwQAejtEMWIlNCng9Xo9RqjF7lSNLG40tiK1B/r29T/0AqcYrSVZB2qddUUDi/ILgndn"
    b"Zylad5yks6Yr44p1wiqdh2iIU9wGCou9Kb9bLZ/8mtNfEpDv+i9fts6XtDSum+3aC2G4wtsCkxmgkWlJsZRhyCPkP86SzGoGaPrD"
    b"SG+AbhclXMrI27X8toDMvtNcsHyukOX0T590SN8Sc01UAl++kSDW7v7L24HmQSDSHK5CyO53oiG96+smBYRHzyP00ksnWu2VBphQ"
    b"eN2J0UM+Oc/pg0VM+YCpd6bnDvJCqT+9sAu6cFjDY5AhHBPvdHL+07ev/745fztycQqCYQswq8KMPqWHZjk+F1B/cYjCeXoZPtP0"
    b"N/r0KM8Eku5yF7NYeEjFccnNTp2zJcwqZjmix2SXvaHGZWP2TZfJ2RB18Yy4SH18RFcerWPrz/nkTU53fZQn/ApWVShx95lCbnwH"
    b"5ez9gz5GIV4xB7MxC46Qct3IiyfkZeDdxV0ztsqvS7e1+f9figoj"
)


def get_system_prompt() -> str:
    try:
        return zlib.decompress(base64.b64decode(_INTERNAL_BLOB)).decode("utf-8")
    except Exception:
        return "Enhance the user's prompt for aesthetic visual generation."
