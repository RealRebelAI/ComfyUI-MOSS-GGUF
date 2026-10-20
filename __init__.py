"""MOSS-Image Rebel: modular dropdown loaders backed by upstream runtime.

Install this folder next to the official moss-image and ComfyUI-GGUF nodes.
No manual path widgets, hidden model copies, or modifications to those plugins.
"""
import folder_paths
from .discovery import options, resolve, inventory
from .runtime import load_transformer, load_encoder, load_vae

CAT = "MOSS-Image/Rebel (GGUF)"

class RebelMossGGUF:
    TITLE = "MOSS-Image GGUF / BF16 Transformer (Rebel)"
    CATEGORY = CAT
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gguf": (options("gguf", folder_paths),),
            "compute_precision": (["bf16", "fp32"], {"default": "bf16"}),
        }}

    def load(self, gguf, compute_precision):
        return (load_transformer(resolve("gguf", gguf, folder_paths), compute_precision),)

class RebelMossEncoder:
    TITLE = "MOSS-Image Qwen3.5 Encoder (Rebel)"
    CATEGORY = CAT
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "encoder": (options("encoder", folder_paths),),
            "device": (["default", "cpu"], {"default": "default"}),
        }}

    def load(self, encoder, device):
        return (load_encoder(resolve("encoder", encoder, folder_paths), device),)

class RebelMossVAE:
    TITLE = "MOSS-Image VAE (Rebel)"
    CATEGORY = CAT
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": (options("vae", folder_paths),)}}

    def load(self, vae):
        return (load_vae(resolve("vae", vae, folder_paths)),)

class RebelMossInventory:
    TITLE = "MOSS Model Inventory (Rebel)"
    CATEGORY = CAT
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("inventory",)
    FUNCTION = "report"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def report(self):
        content = inventory(folder_paths)
        print("\n" + content + "\n")
        return {"ui": {"text": [content]}, "result": (content,)}


class RebelMossLoRA:
    """MOSS-aware LoRA loader for both GGUF and BF16 Rebel MODEL outputs.

    Unlike ComfyUI's generic LoRA loader, converts fused QKV LoRA tensors
    into three adapters that match OpenMOSS's split projection weights.
    """
    TITLE = "MOSS-Image LoRA (Rebel)"
    CATEGORY = CAT
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
        }}

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
        import comfy.utils
        from .moss_lora import apply_moss_lora

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded_lora is None or self.loaded_lora[0] != path:
            lora, metadata = comfy.utils.load_torch_file(
                path, safe_load=True, return_metadata=True
            )
            self.loaded_lora = (path, lora, metadata)
        _, lora, metadata = self.loaded_lora
        return apply_moss_lora(
            model, clip, lora, strength_model, strength_clip, metadata
        )


class RebelMossLoRAModelOnly(RebelMossLoRA):
    TITLE = "MOSS-Image LoRA Model Only (Rebel)"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model_lora"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
        }}

    def load_model_lora(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0.0)[0],)

NODE_CLASS_MAPPINGS = {
    "RebelMossGGUFV4": RebelMossGGUF,
    "RebelMossEncoderV4": RebelMossEncoder,
    "RebelMossVAEV4": RebelMossVAE,
    "RebelMossInventoryV4": RebelMossInventory,
    "RebelMossLoRA": RebelMossLoRA,
    "RebelMossLoRAModelOnly": RebelMossLoRAModelOnly,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
