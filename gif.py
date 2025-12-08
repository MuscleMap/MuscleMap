import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio

# paden
img_path = Path(r"D:\PS_Muscle_Segmentation\Monai\Current_projects\MuscleMap\archive\Final_MuscleMap\githubimages\github.nii.gz")
seg_path = Path(r"D:\PS_Muscle_Segmentation\Monai\Current_projects\MuscleMap\archive\Final_MuscleMap\githubimages\github_dseg.nii.gz")
out_dir = Path(r"D:\PS_Muscle_Segmentation\Monai\Current_projects\MuscleMap\archive\Final_MuscleMap\githubimages\gif")

out_dir.mkdir(parents=True, exist_ok=True)

# NIfTI inladen
img_nii = nib.load(img_path)
seg_nii = nib.load(seg_path)

img = img_nii.get_fdata()
seg = seg_nii.get_fdata()

if img.shape != seg.shape:
    raise ValueError(f"Shapes differ: img {img.shape}, seg {seg.shape}")

print("Volume shape:", img.shape)

# we nemen aan dat de slices over de laatste as gaan (H, W, S)
axis = 2
n_slices = img.shape[axis]

# elke 5e slice
slice_indices = list(range(0, n_slices, 5))
print("Using slices:", slice_indices)

frames = []

# vaste kleurenlijst voor labels (R, G, B, A)
label_colors = [
    (255,   0,   0, 90),  # rood
    (  0, 255,   0, 90),  # groen
    (  0,   0, 255, 90),  # blauw
    (255, 255,   0, 90),  # geel
    (255,   0, 255, 90),  # magenta
    (  0, 255, 255, 90),  # cyaan
    (255, 128,   0, 90),  # oranje
    (128,   0, 255, 90),  # paars
    (  0, 128, 255, 90),  # lichtblauw
]

# ---- 1x: alle labels in hele volume, excl. achtergrond ----
all_labels = np.unique(seg)
all_labels = all_labels[all_labels > 0]
all_labels = np.sort(all_labels)

# vaste mapping: labelwaarde -> kleur
color_map = {}
for i, lab in enumerate(all_labels):
    color_map[lab] = label_colors[i % len(label_colors)]

for idx in slice_indices:
    # slice pakken
    if axis == 0:
        img_slice = img[idx, :, :]
        seg_slice = seg[idx, :, :]
    elif axis == 1:
        img_slice = img[:, idx, :]
        seg_slice = seg[:, idx, :]
    else:  # axis == 2
        img_slice = img[:, :, idx]
        seg_slice = seg[:, :, idx]

    # MRI normaliseren naar 0–255
    sl = img_slice.astype(np.float32)
    sl -= sl.min()
    if sl.max() > 0:
        sl /= sl.max()
    sl_uint8 = (sl * 255).astype(np.uint8)

    # basisbeeld (grijs)
    base_img = Image.fromarray(sl_uint8).convert("RGB")

    # ---- overlay met vaste kleuren per label ----
    overlay_arr = np.zeros((seg_slice.shape[0], seg_slice.shape[1], 4), dtype=np.uint8)

    for lab, color in color_map.items():
        overlay_arr[seg_slice == lab] = color

    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    base_rgba = base_img.convert("RGBA")
    composited = Image.alpha_composite(base_rgba, overlay)

    # 90° met de klok mee draaien
    composited = composited.transpose(Image.ROTATE_270)

    # PNG per slice
    frame_path = out_dir / f"slice_{idx:04d}.png"
    composited.save(frame_path)

    # voor GIF
    frames.append(composited.convert("P"))

# GIF maken
gif_path = out_dir / "musclemap_scroll.gif"
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=80,
    loop=0,
)

print("Saved GIF to:", gif_path)
