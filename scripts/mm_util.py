import os
import logging
import sys
import json
import math
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from monai.transforms import (MapTransform)
import gc, torch
import os, gc, torch, nibabel as nib
import shutil
import psutil
from scipy import ndimage as ndi
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import pandas as pd

AUTO_CHUNK_GPU_SAFETY_MARGIN = 0.70
AUTO_CHUNK_CPU_SAFETY_MARGIN = 0.35
AUTO_CHUNK_GPU_MIN_RESERVE_BYTES = 1.5 * 1024**3
AUTO_CHUNK_CPU_MIN_RESERVE_BYTES = 4 * 1024**3
AUTO_CHUNK_GPU_ESTIMATE_OVERHEAD = 2.50
AUTO_CHUNK_CPU_ESTIMATE_OVERHEAD = 2.00
AUTO_CHUNK_CPU_MAX_LOGIT_BYTES = 2 * 1024**3
AUTO_CHUNK_CPU_LOGIT_FRACTION = 0.25

#check_image_exists 
def check_image_exists(image_path):
    if not os.path.isfile(image_path):
        logging.error(f"Image file '{image_path}' does not exist or is not a file.")
        sys.exit(1)
    if not os.access(image_path, os.R_OK):
        logging.error(f"Image file '{image_path}' is not readable.")
        sys.exit(1)

def get_model_and_config_paths(region, specified_model=None):
    models_base_dir = os.path.join(os.path.dirname(__file__), "models", region)
    if specified_model:
        model_path = os.path.join(models_base_dir, specified_model)
        config_path = os.path.splitext(model_path)[0] + ".json"
        if not os.path.isfile(model_path):
            logging.error(f"Specified model '{specified_model}' does not exist.")
            sys.exit(1)
        if not os.path.isfile(config_path):
            logging.error(f"Config file for model '{specified_model}' does not exist.")
            sys.exit(1)
    else:
        if not os.path.isdir(models_base_dir):
            logging.error(f"Region folder '{region}' does not exist.")
            sys.exit(1)
        
        # Assuming only one model file and one config file in each region folder
        model_path = None
        config_path = None

        for file in os.listdir(models_base_dir):
            if file.endswith(".pth"):
                model_path = os.path.join(models_base_dir, file)
            elif file.endswith(".json"):
                config_path = os.path.join(models_base_dir, file)

        if not model_path:
            logging.error(f"No model file found in region folder '{region}'.")
            sys.exit(1)
        if not config_path:
            logging.error(f"No config file found in region folder '{region}'.")
            sys.exit(1)
    return model_path, config_path

def get_template_paths(region, specified_template=None):
    templates_base_dir = os.path.join(os.path.dirname(__file__), "templates", region)
    
    if not os.path.isdir(templates_base_dir):
        logging.error(f"Region folder '{region}' does not exist.")
        sys.exit(1)
    
    print(templates_base_dir)

    if specified_template:
        template_path = os.path.join(templates_base_dir, specified_template + '.nii.gz')
        template_segmentation_path = os.path.join(templates_base_dir, specified_template + '_dseg.nii.gz')
    else:
        template_path = os.path.join(templates_base_dir, region + '_template.nii.gz')
        template_segmentation_path = os.path.join(templates_base_dir, region + '_template_dseg.nii.gz')
        
    
        if not os.path.isfile(template_path):
            logging.error(f"No template file found in region folder '{region}': ${template_path}.")
            sys.exit(1)
            
        if not os.path.isfile(template_segmentation_path):
            logging.error(f"No template segmentation file found in region folder '{region}': ${template_segmentation_path}.")
            sys.exit(1)

    return template_path, template_segmentation_path

def load_model_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Error: The configuration file '{config_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        logging.error(f"Error parsing the configuration file: {exc}")
        sys.exit(1)

def validate_seg_arguments(args):
    required_args = {'input_image': "-i", 'region': "-r"}
    for arg_name, flag in required_args.items():
        arg_value = getattr(args, arg_name, None)
        if not arg_value:
            logging.error(f"Error: The {arg_name} ({flag}) argument is required.")
            sys.exit(1)
        if not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} ({flag}) argument must be a string.")
            sys.exit(1)

    # Optional arguments validation
    optional_args = {'model': "-m"}
    for arg_name, flag in optional_args.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value and not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} ({flag}) argument must be a string.")
            sys.exit(1)

def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ("KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PiB"

def _release_memory(device=None):
    gc.collect()
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

def _is_oom_error(exc: Exception) -> bool:
    current = exc
    while current is not None:
        if isinstance(current, (MemoryError, torch.cuda.OutOfMemoryError)):
            return True
        msg = str(current).lower()
        if any(
            token in msg
            for token in (
                "out of memory",
                "cuda error: out of memory",
                "can't allocate memory",
                "cannot allocate memory",
                "std::bad_alloc",
                "bad alloc",
            )
        ):
            return True
        current = current.__cause__
    return False

def _resolve_target_spacing(header_zooms, target_pixdim):
    if not target_pixdim:
        return tuple(float(z) for z in header_zooms[:3])

    resolved = []
    for axis, zoom in enumerate(header_zooms[:3]):
        try:
            target = float(target_pixdim[axis])
        except (IndexError, TypeError, ValueError):
            target = float(zoom)
        if target <= 0:
            target = float(zoom)
        resolved.append(target)
    return tuple(resolved)

def _read_int_file(path: Path, allow_zero=False):
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound or value >= 1 << 60:
        return None
    return value

def _get_cgroup_memory_budget():
    proc_cgroup = Path("/proc/self/cgroup")
    candidates = []

    if proc_cgroup.exists():
        try:
            lines = proc_cgroup.read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            _, controllers, relative_path = parts
            rel = relative_path.lstrip("/")

            if controllers == "":
                base = Path("/sys/fs/cgroup")
                scoped = base / rel if rel else base
                candidates.extend(
                    [
                        (scoped / "memory.max", scoped / "memory.current"),
                        (base / "memory.max", base / "memory.current"),
                    ]
                )
                continue

            if "memory" not in controllers.split(","):
                continue

            base = Path("/sys/fs/cgroup/memory")
            scoped = base / rel if rel else base
            candidates.extend(
                [
                    (scoped / "memory.limit_in_bytes", scoped / "memory.usage_in_bytes"),
                    (base / "memory.limit_in_bytes", base / "memory.usage_in_bytes"),
                ]
            )

    candidates.extend(
        [
            (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
            (
                Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            ),
        ]
    )

    seen = set()
    for limit_path, current_path in candidates:
        key = (str(limit_path), str(current_path))
        if key in seen:
            continue
        seen.add(key)

        limit_bytes = _read_int_file(limit_path)
        current_bytes = _read_int_file(current_path, allow_zero=True)
        if limit_bytes is None or current_bytes is None:
            continue

        return {
            "limit_bytes": limit_bytes,
            "current_bytes": current_bytes,
            "available_bytes": max(limit_bytes - current_bytes, 0),
            "source": str(limit_path.parent),
        }

    return None

def _get_system_memory_budget():
    mem = psutil.virtual_memory()
    free_bytes = int(mem.available)
    total_bytes = int(mem.total)
    source = "system"

    cgroup_budget = _get_cgroup_memory_budget()
    if cgroup_budget is not None:
        free_bytes = min(free_bytes, int(cgroup_budget["available_bytes"]))
        total_bytes = min(total_bytes, int(cgroup_budget["limit_bytes"]))
        source = f"system+cgroup:{cgroup_budget['source']}"

    return free_bytes, total_bytes, source

def estimate_auto_chunk_size(image_path, device, out_channels=None, target_pixdim=None):
    img_nii = nib.load(image_path)
    header = img_nii.header
    dims = header.get_data_shape()[:3]
    zooms = header.get_zooms()[:3]
    del img_nii

    if len(dims) < 3 or dims[-1] < 1:
        return 1

    _release_memory(device)

    system_free_bytes, system_total_bytes, system_source = _get_system_memory_budget()
    system_reserve_bytes = max(AUTO_CHUNK_CPU_MIN_RESERVE_BYTES, int(system_total_bytes * 0.10))
    system_usable_bytes = int(max(system_free_bytes - system_reserve_bytes, 0) * AUTO_CHUNK_CPU_SAFETY_MARGIN)

    memory_source = system_source
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        reserve_bytes = max(AUTO_CHUNK_GPU_MIN_RESERVE_BYTES, int(total_bytes * 0.10))
        safety_margin = AUTO_CHUNK_GPU_SAFETY_MARGIN
        overhead_factor = AUTO_CHUNK_GPU_ESTIMATE_OVERHEAD
        memory_source = f"cuda:{device_index}"
    else:
        free_bytes = system_free_bytes
        total_bytes = system_total_bytes
        reserve_bytes = system_reserve_bytes
        safety_margin = AUTO_CHUNK_CPU_SAFETY_MARGIN
        overhead_factor = AUTO_CHUNK_CPU_ESTIMATE_OVERHEAD

    usable_bytes = int(max(free_bytes - reserve_bytes, 0) * safety_margin)
    if usable_bytes <= 0:
        logging.warning(
            "Auto chunk sizing found no free headroom after reserves; falling back to 1 slice "
            f"(free={_format_bytes(free_bytes)} reserve={_format_bytes(reserve_bytes)})."
        )
        return 1

    resolved_spacing = _resolve_target_spacing(zooms, target_pixdim)
    resampled_dims = [
        max(1, int(math.ceil(float(dim) * float(zoom) / float(spacing))))
        for dim, zoom, spacing in zip(dims, zooms, resolved_spacing)
    ]

    out_channels = max(int(out_channels or 1), 1)
    f32 = np.dtype(np.float32).itemsize  # 4

    depth_scale = max(float(zooms[2]) / float(resolved_spacing[2]), 1e-6)

    # Memory at resampled resolution (inference): input tensor + logits
    resampled_bytes_per_slice = int(math.ceil(
        resampled_dims[0] * resampled_dims[1] * depth_scale
        * (f32 + out_channels * f32)  # input + logits
    ))

    # Memory at original resolution (post-processing inverse resample):
    # affine grid (3 floats) + resampled output (out_channels floats)
    original_bytes_per_slice = int(math.ceil(
        dims[0] * dims[1] * 1.0  # original voxels per input slice (no depth scaling)
        * (3 * f32 + out_channels * f32)  # grid + output
    ))

    bytes_per_input_slice = int(math.ceil(
        (resampled_bytes_per_slice + original_bytes_per_slice) * overhead_factor
    ))

    estimated_chunk = max(1, min(int(dims[2]), usable_bytes // max(bytes_per_input_slice, 1)))

    # On CPU, post-processing competes for the same system RAM — apply extra caps.
    # On GPU, post-processing stays on the device and the overhead factor already covers it.
    system_cap_chunk = None
    logit_cap_chunk = None
    if not (device is not None and device.type == "cuda" and torch.cuda.is_available()):
        cpu_bytes_per_input_slice = int(math.ceil(
            (resampled_bytes_per_slice + original_bytes_per_slice)
            * AUTO_CHUNK_CPU_ESTIMATE_OVERHEAD
        ))
        system_cap_chunk = max(1, system_usable_bytes // max(cpu_bytes_per_input_slice, 1))

        logit_bytes_per_input_slice = int(math.ceil(
            resampled_dims[0] * resampled_dims[1] * depth_scale
            * out_channels * f32
        ))
        max_logit_bytes = min(int(system_usable_bytes * AUTO_CHUNK_CPU_LOGIT_FRACTION), AUTO_CHUNK_CPU_MAX_LOGIT_BYTES)
        logit_cap_chunk = max(1, max_logit_bytes // max(logit_bytes_per_input_slice, 1))

        estimated_chunk = min(estimated_chunk, system_cap_chunk, logit_cap_chunk)

    logging.info(
        "Auto chunk sizing: free=%s usable=%s reserve=%s estimated=%s slice(s)(source=%s,overhead=%.2f%s).",
        _format_bytes(free_bytes),
        _format_bytes(usable_bytes),
        _format_bytes(reserve_bytes),
        estimated_chunk,
        memory_source,
        overhead_factor,
        (f", cpu_cap={system_cap_chunk}, logit_cap={logit_cap_chunk}") if system_cap_chunk is not None else "",
    )
    return estimated_chunk

def _run_inference_on_file(
    image_path,
    pre_transforms,
    post_transforms,
    amp_context,
    device,
    inferer,
    model,
):
    data = None
    tensor = None
    pred = None
    single_pred = None
    post_in = None
    post_out = None
    seg_tensor = None
    try:
        data = {"image": image_path}
        data = pre_transforms(data)
        tensor = data["image"]
        if device.type == "cpu":
            tensor = tensor.float()
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device, non_blocking=device.type == "cuda")

        with amp_context, torch.inference_mode():
            pred = inferer(tensor, model)

        single_pred = pred.squeeze(0).squeeze(0)
        del pred
        pred = None
        post_in = {
            "pred": single_pred,
            "image": data["image"],
            "image_meta_dict": data["image_meta_dict"],
        }
        post_out = post_transforms(post_in)
        seg_tensor = post_out["pred"].detach().cpu().to(torch.int16)
        return seg_tensor.numpy().copy()
    finally:
        del data, tensor, pred, single_pred, post_in, post_out, seg_tensor
        _release_memory(device)

def _write_temp_chunk(image_proxy, affine, header, temp_dir, start, end):
    vol_chunk = np.asarray(image_proxy.dataobj[..., start:end], dtype=np.float32)
    chunk_path = os.path.join(temp_dir, f"chunk_{start}_{end}.nii")
    nib.save(nib.Nifti1Image(vol_chunk, affine, header.copy()), chunk_path)
    del vol_chunk
    return chunk_path

def save_nifti(data: np.ndarray, affine, header, out_path):
    new_hdr = header.copy()                            
    img = nib.Nifti1Image(data, affine, new_hdr)
    
    _, qcode = header.get_qform(coded=True)
    _, scode = header.get_sform(coded=True)
    img.set_qform(affine, int(qcode))
    img.set_sform(affine, int(scode))
    nib.save(img, out_path)

def validate_extract_args(args):
    if args.method == 'dixon':
        if not args.fat_image or not args.water_image or not args.segmentation_image:
            print("For dixon method, you must provide -f (fat image), -w (water image), and -s (segmentation image).")
            exit(1)
    elif args.method in ['kmeans', 'gmm']:
        if not args.input_image or not args.components or not args.segmentation_image:
            print("For kmeans or gmm method, you must provide -i (input image), -c (number of components), and -s (segmentation image).")
            exit(1)
    elif args.method == 'average':
        if not args.input_image or not args.segmentation_image:
            print("For average, you must provide -i (input image) and -s (segmentation image).")
    string_args = ['fat_image', 'water_image', 'segmentation_image', 'input_image', 'region', 'output_dir']
    for arg_name in string_args:
        arg_value = getattr(args, arg_name, None)
        if arg_value and not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} argument must be a string.")
            sys.exit(1)

def validate_register_to_template_args(args):
    if not args.input_image or not args.components or not args.segmentation_image:
        print("You must provide -i (input image), -s (segmentation image), and -r (region).")
        exit(1)
    string_args = ['input_image', 'segmentation_image', 'region', 'output_dir']
    for arg_name in string_args:
        arg_value = getattr(args, arg_name, None)
        if arg_value and not isinstance(arg_value, str):
            logging.error(f"Error: The {arg_name} argument must be a string.")
            sys.exit(1)

def extract_image_data(image_path):
    img = nib.load(image_path)
    img_array = img.get_fdata()
    
    dim_x, dim_y, dim_z = img.header['dim'][1:4] #dim_z = number of axial slices
    pixdim_x, pixdim_y, pixdim_z = img.header['pixdim'][1:4] #voxel dimensions in mm
    affine = img.affine
    header = img.header
    
    return img, img_array,affine,header,(dim_x, dim_y, dim_z), (pixdim_x, pixdim_y, pixdim_z)

def add_slice_counts(
    results_entry: Dict[int, Dict[str, Any]],
    label_img:     np.ndarray,
    pix_dim:       Tuple[float, float, float],
    col_name:      str = "Slices with segmentation",
) -> Dict[int, Dict[str, Any]]:
    
    if label_img.ndim != 3:
        raise ValueError("label_img must be 3-D")
    
    pix_dim = tuple(float(p) for p in pix_dim)
    max_axis = int(np.argmax(pix_dim))                
    if max(pix_dim) / min(pix_dim) < 1.01:           
        max_axis = 2                                
    axes_to_reduce = tuple(ax for ax in range(3) if ax != max_axis)

    for lbl in np.unique(label_img):
        lbl = int(lbl)
        if lbl == 0:
            continue
        slice_present = np.any(label_img == lbl, axis=axes_to_reduce)  
        slice_count   = int(slice_present.sum())

        entry = results_entry.get(lbl, {"Label": lbl, "Anatomy": ""})
        entry[col_name] = slice_count
        results_entry[lbl] = entry

    return results_entry

def apply_clustering(args, mask_img, components): 
    if args.method == 'kmeans':
        clustering = KMeans(n_clusters = components, init = 'k-means++', tol = 0.001, n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.labels_ 
    elif args.method == 'gmm':
        clustering = GaussianMixture(n_components = components, covariance_type = 'full', init_params = 'kmeans', tol = 0.001, n_init = 20, max_iter = 1000).fit(mask_img)
        labels = clustering.predict(mask_img)
    else:
        raise ValueError("Either KMeans or GMM must be activated.")
    return labels, clustering

def calculate_thresholds(labels, mask_img, num_clusters):
    clusters = [mask_img[labels == i] for i in range(num_clusters)]
    means = [np.mean(cluster) for cluster in clusters]
    if num_clusters == 2:
        muscle_max = np.max(clusters[0]) if means[0] < means[1] else np.max(clusters[1])
        muscle_img = mask_img[mask_img <= muscle_max]
        fat_min = None # placeholder
        sorted_indices = [0, 1] if means[0] < means[1] else [1, 0]
    elif num_clusters == 3:
        sorted_clusters = sorted(zip(means, clusters, range(len(clusters))), key=lambda x: x[0])
        muscle_img = sorted_clusters[0][1]
        fat_img = sorted_clusters[2][1]
        muscle_max = np.max(muscle_img)
        fat_min= np.min(fat_img)
        sorted_indices = [x[2] for x in sorted_clusters]
    return muscle_max, fat_min, sorted_indices

def create_image_array(img_array, mask_array, label, muscle_upper, fat_lower, components):
    if components not in (2, 3):
        raise ValueError("components must be 2 or 3")

    muscle_label = (mask_array == label) 
    if components == 2:
        muscle_array    = muscle_label & (img_array <  muscle_upper)
        fat_array       = muscle_label & (img_array >= muscle_upper)
        undefined_array = np.zeros_like(img_array, dtype=bool)  # placeholder
    else:  # components == 3
        muscle_array    = muscle_label & (img_array <  muscle_upper)
        undefined_array = muscle_label & (img_array >= muscle_upper) & (img_array < fat_lower)
        fat_array       = muscle_label & (img_array >= fat_lower)
    return muscle_array, fat_array, undefined_array

def create_output_dir(output_dir=None):
    if not output_dir:
        output_dir = os.getcwd()  # Use the current working directory if no output directory is provided
    else:
        # Construct the path to the output directory from the current working directory
        output_dir = os.path.abspath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Output directory {output_dir} created")
    return output_dir

def build_entry_dict_metrics(
    label_img: np.ndarray,
    model_config: Optional[dict],
    region: bool = False,
) -> Dict[int, Dict[str, Any]]:
    
    results_entry: Dict[int, Dict[str, Any]] = {}

    model_labels = (model_config.get("labels", []) if model_config else [])
    idx: Dict[int, str] = {}

    if region and model_labels:
        for L in model_labels:
            try:
                val = int(L.get("value"))
            except Exception:
                continue
            anatomy = str(L.get("anatomy", "")).strip()
            side    = str(L.get("side", "")).strip()
            text = f"{anatomy} {side}".strip()
            if text:
                idx[val] = text

    unmatched_labels: list[int] = []
    for lbl in np.unique(label_img):
        lbl = int(lbl)
        if lbl <= 0:
            continue

        anatomy_text = idx.get(lbl, "") if (region and idx) else ""
        if region and idx and anatomy_text == "":
            unmatched_labels.append(lbl)

        results_entry[lbl] = {
            "Anatomy": anatomy_text,
            "Label":   lbl,
        }

    if region and idx and unmatched_labels:
        logging.warning(
            "No MuscleMap anatomy-side mapping was found for the following label IDs in "
            "the current region configuration: %s. Only label numbers will be given",
            ", ".join(map(str, unmatched_labels))
        )

    return results_entry

def calculate_metrics_dixon(
    result_entry: Dict[int, Dict[str, Any]], 
    label_img: np.ndarray,
    fat_array: np.ndarray,
    water_array: np.ndarray,
    pix_dim: Tuple[float, float, float],
) -> Dict[int, Dict[str, Any]]:
    
    # raise value error when shapes do no match or when 4D is given as input
    if not (label_img.shape == water_array.shape == fat_array.shape):
        raise ValueError("Shape mismatch: segmentation image, water image, and fat image must have identical shapes.")
    if len(pix_dim) != 3:
        raise ValueError("pix_dim must be a 3-tuple (mm, mm, mm)")
    
    # fix voxel_vol_ml to calculate volume in ml
    voxel_vol_ml = (pix_dim[0] * pix_dim[1] * pix_dim[2]) / 1000.0

    # 1) Creating total fat fraction map for formula fat_signal/fat_signal + water signal. 
    denom = fat_array + water_array
    ff_map = np.divide(
        fat_array, denom,
        out=np.zeros_like(denom, dtype=np.float32),
        where=(denom != 0)
    ).astype(np.float32)

    # 2) Flatten voor aggregration and set to int64 for efficiency
    flat_ff  = ff_map.ravel()
    flat_lbl = label_img.astype(np.int64).ravel()

    # 3) get max label from image. Labels in entry but not in image will be skipped.
    max_label = int(flat_lbl.max()) if flat_lbl.size else 0

    # 4) Sum and count per label
    sum_per_lbl   = np.bincount(flat_lbl, weights=flat_ff, minlength=max_label + 1)
    count_per_lbl = np.bincount(flat_lbl, minlength=max_label + 1)

    # 5) mean per label
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_per_lbl = np.divide(
            sum_per_lbl, count_per_lbl,
            out=np.zeros_like(sum_per_lbl, dtype=np.float64),
            where=(count_per_lbl != 0)
        )
    # 6) Update result_entry with fat percentages for each label
    for _k, entry in result_entry.items():
        lbl = int(entry.get("Label", _k))  
        if lbl == 0:
            continue
        if lbl <= max_label and count_per_lbl[lbl] > 0:
            fat_pct = round(float(mean_per_lbl[lbl] * 100.0), 2)
            vol_ml  = round(float(count_per_lbl[lbl] * voxel_vol_ml), 2)
        else:
            fat_pct = np.nan
            vol_ml  = np.nan
        entry.update({
            "Fat (%)":     fat_pct,
            "Volume (ml)": vol_ml,
        })
    # 7) Return updated result_entry dictionary
    return result_entry

def calculate_metrics_average(
    result_entry: Dict[int, Dict[str, Any]],
    label_img: np.ndarray,
    img_array: np.ndarray,
    pix_dim: Tuple[float, float, float],
) -> Dict[int, Dict[str, Any]]:
    
    #Raise ValueError when mismatch or image not in 3D
    if label_img.shape != img_array.shape:
        raise ValueError("Shape mismatch: Segmentation image and img_array must have the same shape")
    if len(pix_dim) != 3:
        raise ValueError("pix_dim must be a 3-tuple (mm, mm, mm)")
    
    # fix voxel_vol_ml to calculate volume in ml 
    voxel_vol_ml = (pix_dim[0] * pix_dim[1] * pix_dim[2]) / 1000.0

    # Vectorized aggregations
    flat_lbl = label_img.astype(np.int64).ravel()
    flat_val = img_array.astype(np.float64).ravel()
    max_label = int(flat_lbl.max()) if flat_lbl.size else 0
    
    # Vectorized calculations for sum and count
    sum_per_lbl   = np.bincount(flat_lbl, weights=flat_val, minlength=max_label + 1)
    count_per_lbl = np.bincount(flat_lbl, minlength=max_label + 1)

    # ignore dividing by zero error and only divide where count_per_lbl is > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_per_lbl = np.divide(
            sum_per_lbl, count_per_lbl,
            out=np.zeros_like(sum_per_lbl, dtype=np.float64),
            where=(count_per_lbl != 0)
        )
    labels_present = np.flatnonzero(count_per_lbl)
    for lbl in labels_present:
        if lbl == 0:
            continue
        avg_int = round(float(mean_per_lbl[lbl]), 2)
        vol_ml  = round(float(count_per_lbl[lbl] * voxel_vol_ml), 2)

        entry = result_entry.get(int(lbl), {"Anatomy": "", "Label": int(lbl)})
        entry.update({
            "Average Intensity": avg_int,
            "Volume (ml)":       vol_ml,
        })
        result_entry[int(lbl)] = entry

    for _k, entry in list(result_entry.items()):
        lbl = int(entry.get("Label", _k))  
        if lbl == 0:
            continue
        if lbl > max_label or count_per_lbl[lbl] == 0:
            entry.setdefault("Average Intensity", np.nan)
            entry.setdefault("Volume (ml)",       np.nan)

    return result_entry

def calculate_metrics_thresholding(
    args,
    results_entry: Dict[str, Any],                            
    label_img: np.ndarray,                
    img_array: np.ndarray,                 
    affine: np.ndarray,
    header:  np.ndarray,                   
    pix_dim: Tuple[float, float, float],  
    components: int,                      
    output_dir: Union[str, Path],          
    id_part: str = "",                   
) -> Dict[str, Any]:
    
    # raise value errors if components is not 2/3 or when mismatch in shape
    if components not in (2, 3):
        raise ValueError("components must be 2 or 3")
    if label_img.shape != img_array.shape:
        raise ValueError("label_img and img_array must have the same shape")
    
    #prepare empty image array to build up the fat, muscle (and in 3 component; undefined)maps    
    total_muscle_image    = np.zeros_like(img_array, dtype=bool)
    total_fat_image       = np.zeros_like(img_array, dtype=bool)
    total_undefined_image = np.zeros_like(img_array, dtype=bool)
    combined_mask = np.zeros_like(label_img, dtype=np.uint8)  
    
    # if GMM is chosen, we will also create an empty array in float32 for each component to store softprob. 
    if args.method == 'gmm':
        total_probability_maps = [np.zeros(label_img.shape, dtype=np.float32)
                                for _ in range(components)]
        
    # determine voxel vol ml to easily calculate volume from pixdim  
    voxel_vol_ml = (pix_dim[0] * pix_dim[1] * pix_dim[2]) / 1000.0  

    # build up the dictionary for 2 or 3 clusters and apply
    for lbl in np.unique(label_img):
        # we will skip background (label == 0 in each model)
        if lbl == 0:
            continue

        #create mask specific the voxels from foreground label
        mask = (label_img == lbl)

        #reshape mask so that it can be used for thresholding (1D)
        mask_img = img_array[mask].reshape(-1, 1)

        # apply the clustering function and we get two (or three) maps with voxels
        labels, clustering = apply_clustering(args,
            mask_img, components
        )

        # calculate thresholds from clustering 
        muscle_max, fat_min, sorted_indices = calculate_thresholds(labels, mask_img, components)
        
        # determine the number of voxels over the 1D vector
        N = mask_img.size

        # determine the total_volume for the label
        total_volume = N * voxel_vol_ml

        # use thresholds (muscle max for bimodal and muscle_max + fat_min for trimodal) to build up image and calculate percentage
        if components == 2:
            # iteravilely build up boolean fat and muscle maps
            muscle_array, fat_array, _ = create_image_array(img_array, label_img, lbl, muscle_max, fat_min, components)
            total_muscle_image |= muscle_array
            total_fat_image |= fat_array
            combined_mask[muscle_array] = 1
            combined_mask[fat_array]    = 4

            # fat and muscle calculations
            muscle_percentage = 100.0 * np.mean((mask_img.ravel() <= muscle_max))
            fat_percentage = 100 - muscle_percentage
            
            # volume calculations
            muscle_voxels = np.count_nonzero(mask_img <= muscle_max)
            muscle_volume = muscle_voxels * voxel_vol_ml
            fat_volume = (N - muscle_voxels) * voxel_vol_ml 

        if components == 3: 
            # iteraively build up fat and muscle maps
            muscle_array, fat_array, undefined_array = create_image_array(img_array, label_img, lbl, muscle_max, fat_min, components)
            total_muscle_image |= muscle_array
            total_fat_image |= fat_array  
            total_undefined_image |= undefined_array

            combined_mask[muscle_array]    = 1   
            combined_mask[undefined_array] = 7 
            combined_mask[fat_array]       = 4  

            #fat,muscle and undefined calculations
            muscle_percentage    = np.nan if N == 0 else 100.0 * np.mean(mask_img <  muscle_max)
            undefined_percentage = np.nan if N == 0 else 100.0 * np.mean((mask_img >= muscle_max) & (mask_img < fat_min))
            fat_percentage       = np.nan if N == 0 else 100.0 * np.mean(mask_img >= fat_min)

            # volume calculations
            muscle_voxels = np.count_nonzero(mask_img <= muscle_max)
            muscle_volume = muscle_voxels * voxel_vol_ml
            undefined_voxels = np.count_nonzero((mask_img > muscle_max) & (mask_img < fat_min))
            undefined_volume = undefined_voxels * voxel_vol_ml
            fat_voxels = np.count_nonzero(mask_img >= fat_min)
            fat_volume = fat_voxels * voxel_vol_ml

        entry = results_entry.get(int(lbl), {"Anatomy": "", "Label": int(lbl)})
        entry.update({
            "Muscle (%)":         (np.nan if muscle_percentage is None else round(float(muscle_percentage), 2)),
            "Fat (%)":            (np.nan if fat_percentage is None else round(float(fat_percentage), 2)),
            "Total volume (ml)":  (np.nan if total_volume is None else round(float(total_volume), 2)),
            "Fat volume (ml)":    (np.nan if fat_volume is None else round(float(fat_volume), 2)),
            "Muscle volume (ml)": (np.nan if muscle_volume is None else round(float(muscle_volume), 2)),
        })

        if components == 3:
            entry["Undefined (%)"]         = (np.nan if undefined_percentage is None else round(float(undefined_percentage), 2))
            entry["Undefined volume (ml)"] = (np.nan if undefined_volume     is None else round(float(undefined_volume),     2))

        results_entry[int(lbl)] = entry

        if args.method == 'gmm':
            probability_maps = clustering.predict_proba(mask_img)             
            sorted_probability_maps = probability_maps[:, sorted_indices]      
            for comp_idx in range(components):
                 total_probability_maps[comp_idx][mask] += sorted_probability_maps[:, comp_idx].astype(np.float32)

    save_nifti(total_muscle_image.astype(np.uint8), affine,header, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_muscle_seg.nii.gz'))
    save_nifti(combined_mask, affine, header,
           os.path.join(output_dir,
                        f"{id_part}_{args.method}_{components}component_combined_seg.nii.gz"))
    if components == 3:
        save_nifti(total_undefined_image.astype(np.uint8), affine,header, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_undefined_seg.nii.gz'))
    save_nifti(total_fat_image.astype(np.uint8), affine,header, os.path.join(output_dir, f'{id_part}_{args.method}_{args.components}component_fat_seg.nii.gz'))
    
    if args.method == 'gmm':
        if components == 3:
            component_names = ["muscle", "undefined", "fat"]
        else:
            component_names = ["muscle", "fat"]
        for comp_idx, comp_name in enumerate(component_names):
            out_path = os.path.join(
                output_dir,
                f"{id_part}_gmm_{comp_name}_{components}component_softseg.nii.gz"
            )
        save_nifti(total_probability_maps[comp_idx], affine, header, out_path)
    return results_entry

def results_entry_to_dataframe(results_entry: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for lbl, entry in results_entry.items():
        label_val = int(entry.get("Label", lbl))
        row = {"Label": label_val}
        row.update(entry)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "Label" in df.columns:
        df = df.drop_duplicates(subset=["Label"]).sort_values("Label")   
    return df

def absolute_path(relative_path):
    base_path = os.path.dirname(__file__)  # Gets the directory where the script is located
    return os.path.join(base_path, relative_path)

class RemapLabels(MapTransform):
    def __init__(self, keys, id_map, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.id_map = id_map
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lab = d[key]
            out = lab.clone()
            for orig, tgt in self.id_map.items():
                out[lab == orig] = tgt
            d[key] = out
        return d
    
class SqueezeTransform(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lab = d[key]
            out = lab.squeeze(0) if lab.dim() > 3 and lab.shape[0] == 1 else lab  # Remove channel dim if [1, H, W, D]
            d[key] = out
        return d
    
def connected_chunks(
    seg: np.ndarray,
    labels: Optional[np.ndarray] = None,
    connectivity: int = 1,  # 3=26-connectivity
) -> np.ndarray:
    
    """
    Keeps only the largest connected component per label in a multi-label segmentation.
    - Supports both 3D (X,Y,Z) and 4D (1,X,Y,Z) arrays.
    - Incorporated to get optimize RAM memory management during inference for large images
    - 
    """
    #Ensure that is nparray
    seg = np.asarray(seg)

    # Add channel dimension if necessary (to unify shape to 4D)
    remove_dim = False
    if seg.ndim == 3:
        seg_ch = seg[None, ...]
        remove_dim = True
    elif seg.ndim == 4 and seg.shape[0] == 1:
        seg_ch = seg
    else:
        raise ValueError(f"Expected (X,Y,Z) or (1,X,Y,Z), got {seg.shape}")

    # find labels excluding background (0)
    if labels is None:
        labels = np.unique(seg_ch)
    labels = labels[labels != 0]
    if labels.size == 0:
        result = seg_ch.astype(np.int16, copy=False)
        return result[0] if remove_dim else result

    # Extract the 3D volume from channel 0 for processing
    vol = seg_ch[0]

    # Connectivity structure for 3D, rank 3 = 26-connectivity to be not to conversative
    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)

    # Buffers for 3D mask and labels
    mask3d = np.empty(vol.shape, dtype=bool)
    lab3d  = np.empty(vol.shape, dtype=np.int32)

    # Process each label independently on the 3D volume
    for lab_id in labels:
        np.equal(vol, lab_id, out=mask3d)
        if not mask3d.any():
            continue
        # Label connected components on mask3d
        ndi.label(mask3d, structure=structure, output=lab3d)
        max_lab = int(lab3d.max())
        if max_lab <= 1:
            continue

        # Compute sizes and pick the largest component
        counts = np.bincount(lab3d.ravel())
        keep = counts[1:].argmax() + 1
        del counts

        # Zero out everything except the largest component for this label
        np.logical_and(mask3d, lab3d != keep, out=mask3d)
        vol[mask3d] = 0

    # Write back the processed 3D volume into output array
    seg_ch[0] = vol
    # Cleanup
    del mask3d, lab3d

    # Convert to int16 and drop channel dim if needed
    result = seg_ch.astype(np.int16, copy=False)
    if remove_dim:
        result = result[0]
    return result

def is_nifti(path: str) -> bool:
    p = path.lower()
    return p.endswith(".nii.gz") or p.endswith(".nii")

def _make_out_path(image_path, output_dir, tag="_dseg"):
    fname = os.path.basename(image_path)
    if fname.endswith(".nii.gz"):
        base = fname[:-7]
    elif fname.endswith(".nii"):
        base = fname[:-4]
    return os.path.join(output_dir, f"{base}{tag}.nii.gz")

def run_inference(
    image_path,
    output_dir,
    pre_transforms,
    post_transforms,
    amp_context=None,
    chunk_size=25,
    device=None,
    inferer=None,
    model=None,
    out_channels=None,
    target_pixdim=None,
):
    out_path = _make_out_path(image_path, output_dir, "_dseg")
    img_nii = nib.load(image_path)
    affine = img_nii.affine.copy()
    header = img_nii.header.copy()
    dims = header.get_data_shape()
    D = dims[-1]
    auto_chunking = isinstance(chunk_size, str) and chunk_size.lower() == "auto"

    if auto_chunking:
        chunk_size = estimate_auto_chunk_size(
            image_path,
            device,
            out_channels=out_channels,
            target_pixdim=target_pixdim,
        )
    else:
        chunk_size = int(chunk_size)

    chunk_size = max(1, min(chunk_size, D))
    logging.info("Using chunk size: %s%s", chunk_size, " (auto)" if auto_chunking else "")

    temp_dir = os.path.join(output_dir, "temp_chunks")
    full_seg = None
    try:
        if chunk_size >= D:
            try:
                seg_np = _run_inference_on_file(
                    image_path,
                    pre_transforms,
                    post_transforms,
                    amp_context,
                    device,
                    inferer,
                    model,
                )
            except Exception as exc:
                if not (auto_chunking and _is_oom_error(exc) and D > 1):
                    raise
                chunk_size = max(1, D // 2)
                logging.warning(
                    "Auto chunking hit OOM on the full volume; retrying with chunk size %s.",
                    chunk_size,
                )
            else:
                full_seg = connected_chunks(seg_np)
                nib.save(nib.Nifti1Image(full_seg, affine, header), out_path)
                del seg_np
                return out_path

        os.makedirs(temp_dir, exist_ok=True)
        full_seg = np.zeros(dims, dtype=np.int16)
        start = 0
        while start < D:
            end = min(start + chunk_size, D)
            chunk_path = None
            try:
                chunk_path = _write_temp_chunk(img_nii, affine, header, temp_dir, start, end)
                seg_np = _run_inference_on_file(
                    chunk_path,
                    pre_transforms,
                    post_transforms,
                    amp_context,
                    device,
                    inferer,
                    model,
                )
                full_seg[..., start:end] = seg_np
                del seg_np
                start = end
            except Exception as exc:
                if not (auto_chunking and _is_oom_error(exc)):
                    raise
                if chunk_size == 1:
                    raise RuntimeError(
                        "Auto chunking could not find a safe chunk size. Inference still OOMs at 1 slice."
                    ) from exc
                new_chunk_size = max(1, chunk_size // 2)
                logging.warning(
                    "OOM while processing slices %s:%s with chunk size %s; retrying with %s.",
                    start,
                    end,
                    chunk_size,
                    new_chunk_size,
                )
                chunk_size = new_chunk_size
            finally:
                if chunk_path and os.path.exists(chunk_path):
                    os.remove(chunk_path)

        full_seg = connected_chunks(full_seg)
        nib.save(nib.Nifti1Image(full_seg, affine, header), out_path)
        return out_path
    finally:
        del img_nii
        if full_seg is not None:
            del full_seg
        _release_memory(device)
        shutil.rmtree(temp_dir, ignore_errors=True)
