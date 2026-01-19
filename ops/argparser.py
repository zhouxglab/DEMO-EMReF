import argparse
import json
from collections import OrderedDict
from pathlib import Path
import sys

def argparser():
    parser = argparse.ArgumentParser(
        description=(
            "╔════════════════════════════════════╗\n"
            "║       Welcome to DEMO-EMReF        ║\n"
            "║ Cryo-EM / ET Map Refinement Tool   ║\n"
            "╚════════════════════════════════════╝\n\n"
            "This tool refines cryo-EM or ET maps using DEMO-EMReF.\n"
            "Example usage:\n"
            "  ./DEMO-EMReF -F input_map.mrc -o output_map.mrc"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-F', type=str, required=True,
        help=(
            "Path to the input cryo-EM or ET map file (MRC format).\n"
            "Example: /path/to/map.mrc"
        )
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to a JSON configuration file. If not provided, the tool will select a default\n"
            "config based on --mode (HR, MR, ET)."
        )
    )
    parser.add_argument(
        "--mode","-m", type=str, choices=["HR", "MR", "ET"], default=False,
        help=(
            "Inference mode:\n"
            "  HR - High-resolution refinement\n"
            "  MR - Medium-resolution refinement\n"
            "  ET - Electron tomography refinement\n"
            "Default: HR"
        )
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="GPU device ID(s), e.g., '0' or '0,1'. Default is '0'."
    )
    parser.add_argument(
        "--output","-o", type=str, required=True,
        help="Path to the output refined map file (MRC format)."
    )
    parser.add_argument(
        "--contour", type=float, default=0,
        help="Contour level for output map visualization. Default: 0"
    )
    parser.add_argument(
        "--inverse_mask", action='store_true', default=False,
        help="Apply inverse mask during refinement. Default: False"
    )
    parser.add_argument(
        "--mask_map", type=str, default=None,
        help="Optional mask map path to limit refinement region."
    )
    parser.add_argument(
        "--mask_contour", "-c", type=float, default=0.0,
        help="Contour level for mask visualization. Default: 0.0"
    )
    parser.add_argument(
        "--mask_str", "-p", type=str, default=None,
        help="Identifier string for mask selection or labeling."
    )
    parser.add_argument(
        "--interp_back", action='store_true', default=False,
        help="Interpolate output back to the input resolution. Default: False"
    )
    parser.add_argument(
        "--batch_size","-b", type=int, default=8,
        help="Batch size for inference. Default: 8, chosen to reduce GPU memory usage. Users with sufficient VRAM may increase this value."
    )
    parser.add_argument(
        "--stride", "-s",type=int, default=24,
        help="Sliding window stride for inference. Default: 24 to save time, for better performance, you can choose 12"
    )

    # If no arguments are provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    params = vars(args)

    # ===== Map mode to config file =====
    CONFIG_MAP = {
        "HR": "config/DEMO-EMReF_infer_HR.json",
        "MR": "config/DEMO-EMReF_infer_MR.json",
        "ET": "config/DEMO-EMReF_infer_ET.json"
    }

    if params["config"] is None:
        if params["mode"] is None:
            params["mode"] = "HR"
        params["config"] = CONFIG_MAP[params["mode"]]

    # ===== Read JSON config =====
    json_str = ''
    opt_path = params["config"]
    config_dir = Path(opt_path).parent.resolve()
    try:
        with open(opt_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        print(f"Error: Config file '{opt_path}' not found!")
        sys.exit(1)

    # Merge JSON config into params
    for key in opt:
        params[key] = opt[key]

    # Adjust model path
    if "model" in params and isinstance(params["model"], dict):
        model_cfg = params["model"]
        if "path" in model_cfg:
            p = Path(model_cfg["path"])
            if not p.is_absolute():
                model_cfg["path"] = str((config_dir / p).resolve())


    return params

