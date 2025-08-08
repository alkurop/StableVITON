import torch
from safetensors.torch import save_file

ckpt_path = "/Users/alex/Downloads/VITONHD.ckpt"
safe_path = "/Users/alex/Downloads/VITONHD.safetensors"

print("Loading .ckpt (this part will still use RAM once)...")
ckpt = torch.load(ckpt_path, map_location="cpu")  # keep CPU to avoid GPU memory spike

# If it's a Lightning checkpoint, extract "state_dict"
if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

print(f"Saving to safetensors: {safe_path}")
save_file(ckpt, safe_path)

print("✅ Done. You can now load the model from .safetensors without pickle overhead.")
