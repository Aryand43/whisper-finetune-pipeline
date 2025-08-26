from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
import os

dcp_checkpoints = [
    "checkpoints/hearty-bee-7",
    "checkpoints/smart-smoke-5",
    "checkpoints/eager-haze-6",
    "checkpoints/wandering-river-4"
]

output_dir = "converted_checkpoints"
os.makedirs(output_dir, exist_ok=True)

for dcp_path in dcp_checkpoints:
    name = os.path.basename(dcp_path)
    out_path = os.path.join(output_dir, f"{name}.pth")
    print(f"Converting {dcp_path} → {out_path}")
    dcp_to_torch_save(dcp_path, out_path)

print("All conversions done.")
