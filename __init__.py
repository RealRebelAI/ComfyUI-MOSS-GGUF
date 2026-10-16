"""MOSS-Image Rebel: modular dropdown loaders backed by upstream runtime.

Install this folder next to the official moss-image and ComfyUI-GGUF nodes.
No manual path widgets, hidden model copies, or modifications to those plugins.
"""
import folder_paths
from .discovery import options, resolve, inventory
from .runtime import load_gguf, load_encoder, load_vae

CAT = "MOSS-Image/Rebel (GGUF)"

class RebelMossGGUF:
    TITLE = "MOSS-Image GGUF Transformer (Rebel)"
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
        return (load_gguf(resolve("gguf", gguf, folder_paths), compute_precision),)

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

NODE_CLASS_MAPPINGS = {
    "RebelMossGGUFV4": RebelMossGGUF,
    "RebelMossEncoderV4": RebelMossEncoder,
    "RebelMossVAEV4": RebelMossVAE,
    "RebelMossInventoryV4": RebelMossInventory,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
