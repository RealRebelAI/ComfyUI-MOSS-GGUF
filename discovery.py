"""Model-picker discovery for standard ComfyUI folders. No user-entered paths."""
from pathlib import Path
import os

MODELS = {
    "gguf": ("unet", "diffusion_models"),
    "encoder": ("text_encoders", "clip"),
    "vae": ("vae",),
}
EXTENSIONS = {
    "gguf": (".gguf", ".safetensors"),
    "encoder": (".safetensors", ".pt", ".pth"),
    "vae": (".safetensors", ".pt", ".pth"),
}
EMPTY = {
    "gguf": "<NO GGUF FOUND - open MOSS Model Inventory>",
    "encoder": "<NO ENCODER FOUND - open MOSS Model Inventory>",
    "vae": "<NO VAE FOUND - open MOSS Model Inventory>",
}


def roots_for(kind, fp):
    """Use registered paths AND the physical standard model directories.

    ComfyUI versions differ on whether 'unet' is a registered alias. All
    paths are derived from folder_paths and never from a hardcoded D: drive.
    """
    roots = []
    models_dir = Path(fp.models_dir)
    for category in MODELS[kind]:
        registered = []
        if category in getattr(fp, "folder_names_and_paths", {}):
            try:
                registered = fp.get_folder_paths(category)
            except Exception:
                registered = fp.folder_names_and_paths[category][0]
        for directory in list(registered) + [models_dir / category]:
            directory = Path(directory)
            if directory.is_dir():
                try:
                    canonical = directory.resolve()
                except OSError:
                    canonical = directory.absolute()
                if (category, canonical) not in roots:
                    roots.append((category, canonical))
    return roots


def discover(kind, fp):
    """Map readable, unique dropdown labels to absolute files (scan subfolders)."""
    result = {}
    for category, root in roots_for(kind, fp):
        for base, dirs, files in os.walk(root):
            dirs.sort(key=str.casefold)
            for filename in sorted(files, key=str.casefold):
                if not filename.lower().endswith(EXTENSIONS[kind]):
                    continue
                full = Path(base) / filename
                rel = full.relative_to(root).as_posix()
                label = f"{category}/{rel}"
                result.setdefault(label, full)
    return dict(sorted(result.items(), key=lambda item: item[0].casefold()))


def options(kind, fp):
    return list(discover(kind, fp)) or [EMPTY[kind]]


def resolve(kind, selected, fp):
    index = discover(kind, fp)
    if selected in index:
        file = index[selected]
        if file.is_file():
            return str(file)
    directories = "\n".join(f"  {category}: {directory}" for category, directory in roots_for(kind, fp)) or "  (none exist)"
    raise FileNotFoundError(
        f"MOSS {kind} selection not found: {selected!r}\n"
        f"Searched:\n{directories}\n"
        f"Found {len(index)} matching files. Restart ComfyUI after adding files."
    )


def inventory(fp):
    lines = ["MOSS model inventory - registered and standard ComfyUI folders"]
    lines += [f"ComfyUI models directory: {fp.models_dir}"]
    for kind in MODELS:
        lines.append(f"\n[{kind.upper()}] roots:")
        roots = roots_for(kind, fp)
        lines.extend(f"  {cat}: {path}" for cat, path in roots)
        if not roots:
            lines.append("  (no directories exist)")
        matches = discover(kind, fp)
        lines.append(f"Found {len(matches)} file(s):")
        lines.extend(f"  {name}" for name in matches)
    return "\n".join(lines)
