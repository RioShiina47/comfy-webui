"""Qwen-Image-2.1 Prompt Enhancer Chain Injector.

This injector automatically integrates the Qwen-Image-2.1 prompt enhancement model into
the generation workflow:
- Detects whether the task is pure text-to-image (T2I) or involves images (I2I, inpainting,
  outpainting, hires.fix, or multi-reference images).
- Automatically selects the appropriate prompt enhancement checkpoint:
  - T2I: qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors
  - I2I / Reference: qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors
- Handles single or multi-reference inputs: multiple reference images are stitched into
  a structured grid layout via ComfyUI native ImageStitch nodes and scaled via ImageScaleToTotalPixels.
- Supports reasoning mode (thinking=True) with automatic reasoning tag stripping via RegexReplace
  (^.*?</think>\\s*) to prevent thought chain text from contaminating downstream image conditioning.
"""

try:
    from utils.app_utils import ensure_file_downloaded
except ImportError:
    ensure_file_downloaded = None


def create_node(assembler, class_type, title):
    """Create a workflow node dictionary with fallback template generation."""
    try:
        node = assembler._get_node_template(class_type)
    except Exception:
        node = {
            "inputs": {},
            "class_type": class_type,
            "_meta": {"title": title}
        }
    if "_meta" not in node:
        node["_meta"] = {}
    node['_meta']['title'] = title
    return node


def inject(assembler, chain_definition, chain_items):
    """Inject Qwen-Image-2.1 prompt enhancer nodes into the target generation workflow."""
    if not chain_items:
        return

    is_enabled = False
    thinking_enabled = False
    max_length = 512
    for item in chain_items:
        if isinstance(item, dict):
            if item.get("enable", False) or item.get("enabled", False) or item.get("value", False):
                is_enabled = True
                if item.get("thinking", False) in (True, "true", "on", "yes", "1", 1):
                    thinking_enabled = True
                if "max_length" in item and item["max_length"]:
                    try:
                        max_length = int(item["max_length"])
                    except (ValueError, TypeError):
                        pass
        elif item is True or str(item).lower() in ("on", "true", "yes", "1"):
            is_enabled = True
            break

    if not is_enabled:
        return

    target_node_name = chain_definition.get('target_node', 'prompt')
    target_node_id = assembler.node_map.get(target_node_name)

    if not target_node_id or target_node_id not in assembler.workflow:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'TextEncodeQwenImage21':
                target_node_id = node_id
                break

    if not target_node_id or target_node_id not in assembler.workflow:
        print(f"Warning: Target node '{target_node_name}' (TextEncodeQwenImage21) not found for Qwen-Image-2.1 Prompt Enhancer. Skipping.")
        return

    target_node = assembler.workflow[target_node_id]
    original_prompt = target_node.get('inputs', {}).get('prompt', '')

    # 1. Check if any LoadImage nodes exist in the workflow
    load_image_nodes = [
        node_id for node_id, node in assembler.workflow.items()
        if isinstance(node, dict) and node.get('class_type') == 'LoadImage'
    ]
    has_load_image = len(load_image_nodes) > 0

    target_image_conn = None

    if not has_load_image:
        # Pure text-to-image (txt2img) task without reference images
        clip_model_name = "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"
    else:
        # Task with reference images or img2img / inpaint / outpaint / hires.fix
        clip_model_name = "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"

        # Collect image connections
        # Prioritize extracting all images.image_* slots from target_node['inputs'] (injected by reference image injectors)
        indexed_slots = []
        target_inputs = target_node.get('inputs', {})
        for slot_key, slot_val in target_inputs.items():
            if slot_key.startswith('images.image_') and isinstance(slot_val, (list, tuple)):
                try:
                    idx = int(slot_key.split('_')[-1])
                except Exception:
                    idx = 999
                indexed_slots.append((idx, slot_val))

        indexed_slots.sort(key=lambda x: x[0])
        ref_image_conns = [s[1] for s in indexed_slots]

        # If no reference image slots exist (e.g. pure img2img / inpaint / outpaint), retrieve from LoadImage nodes
        if not ref_image_conns:
            for lid in load_image_nodes:
                scaled_conn = None
                for nid, node in assembler.workflow.items():
                    if isinstance(node, dict) and node.get('class_type') == 'ImageScaleToTotalPixels':
                        img_input = node.get('inputs', {}).get('image')
                        if isinstance(img_input, (list, tuple)) and str(img_input[0]) == str(lid):
                            scaled_conn = [nid, 0]
                            break
                if scaled_conn:
                    ref_image_conns.append(scaled_conn)
                else:
                    scale_id = assembler._get_unique_id()
                    scale_node = create_node(assembler, "ImageScaleToTotalPixels", "Scale Input Image")
                    scale_node['inputs'] = {
                        "upscale_method": "nearest-exact",
                        "megapixels": 1,
                        "resolution_steps": 1,
                        "image": [lid, 0]
                    }
                    assembler.workflow[scale_id] = scale_node
                    ref_image_conns.append([scale_id, 0])

        # Organize multiple reference images into grid stitching
        if len(ref_image_conns) == 1:
            target_image_conn = ref_image_conns[0]
        elif len(ref_image_conns) >= 2:
            num_cols = 2 if len(ref_image_conns) <= 4 else 3

            rows = []
            for i in range(0, len(ref_image_conns), num_cols):
                rows.append(ref_image_conns[i:i + num_cols])

            # Horizontally stitch images in each row
            stitched_rows = []
            for r_idx, row_imgs in enumerate(rows):
                if len(row_imgs) == 1:
                    stitched_rows.append(row_imgs[0])
                else:
                    curr_stitch = row_imgs[0]
                    for c_idx in range(1, len(row_imgs)):
                        st_id = assembler._get_unique_id()
                        st_node = create_node(assembler, "ImageStitch", f"Stitch Images (Row {r_idx+1}-{c_idx})")
                        st_node['inputs'] = {
                            "direction": "right",
                            "match_image_size": True,
                            "spacing_width": 0,
                            "spacing_color": "white",
                            "image1": curr_stitch,
                            "image2": row_imgs[c_idx]
                        }
                        assembler.workflow[st_id] = st_node
                        curr_stitch = [st_id, 0]
                    stitched_rows.append(curr_stitch)

            # Vertically stitch multiple rows
            if len(stitched_rows) == 1:
                final_stitched = stitched_rows[0]
            else:
                curr_vertical = stitched_rows[0]
                for r_idx in range(1, len(stitched_rows)):
                    st_id = assembler._get_unique_id()
                    st_node = create_node(assembler, "ImageStitch", f"Stitch Images (Vertical {r_idx})")
                    st_node['inputs'] = {
                        "direction": "down",
                        "match_image_size": True,
                        "spacing_width": 0,
                        "spacing_color": "white",
                        "image1": curr_vertical,
                        "image2": stitched_rows[r_idx]
                    }
                    assembler.workflow[st_id] = st_node
                    curr_vertical = [st_id, 0]
                final_stitched = curr_vertical

            # Scale final stitched image to ~1024 level via ImageScaleToTotalPixels
            final_scale_id = assembler._get_unique_id()
            final_scale_node = create_node(assembler, "ImageScaleToTotalPixels", "Scale Stitched Image")
            final_scale_node['inputs'] = {
                "upscale_method": "nearest-exact",
                "megapixels": 1,
                "resolution_steps": 1,
                "image": final_stitched
            }
            assembler.workflow[final_scale_id] = final_scale_node
            target_image_conn = [final_scale_id, 0]

    # 2. Ensure model is downloaded in CPU stage before workflow execution
    if ensure_file_downloaded:
        try:
            ensure_file_downloaded(clip_model_name)
        except Exception as e:
            print(f"Warning: Failed to ensure '{clip_model_name}' downloaded: {e}")

    # 2. Create CLIPLoader node
    clip_node_id = assembler._get_unique_id()
    clip_node = create_node(assembler, "CLIPLoader", "Load CLIP")
    clip_node['inputs'] = {
        "clip_name": clip_model_name,
        "type": "qwen_image",
        "device": "default"
    }
    assembler.workflow[clip_node_id] = clip_node

    # 3. Create TextGenerate node
    text_gen_node_id = assembler._get_unique_id()
    text_gen_node = create_node(assembler, "TextGenerate", "Generate Text")
    text_gen_inputs = {
        "prompt": original_prompt,
        "max_length": max_length,
        "sampling_mode": "on",
        "sampling_mode.temperature": 0.7,
        "sampling_mode.top_k": 20,
        "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0.0,
        "sampling_mode.repetition_penalty": 1.05,
        "sampling_mode.seed": 0,
        "sampling_mode.presence_penalty": 0.0 if has_load_image else 1.5,
        "thinking": thinking_enabled,
        "use_default_template": True,
        "mtp": "auto",
        "clip": [clip_node_id, 0]
    }
    if target_image_conn is not None:
        text_gen_inputs["image"] = target_image_conn

    text_gen_node['inputs'] = text_gen_inputs
    assembler.workflow[text_gen_node_id] = text_gen_node

    # 4. Insert Replace Text (Regex) node if thinking is enabled to strip reasoning tags
    if thinking_enabled:
        regex_node_id = assembler._get_unique_id()
        regex_node = create_node(assembler, "RegexReplace", "Replace Text (Regex)")
        regex_node['inputs'] = {
            "string": [text_gen_node_id, 0],
            "regex_pattern": "^.*?</think>\\s*",
            "replace": "",
            "case_insensitive": True,
            "multiline": False,
            "dotall": True,
            "count": 0
        }
        assembler.workflow[regex_node_id] = regex_node
        target_node['inputs']['prompt'] = [regex_node_id, 0]
        print(f"[Injector] Qwen-Image-2.1 Prompt Enhancer applied successfully with RegexReplace ({'I2I' if has_load_image else 'T2I'}, clip='{clip_model_name}', thinking=True).")
    else:
        target_node['inputs']['prompt'] = [text_gen_node_id, 0]
        print(f"[Injector] Qwen-Image-2.1 Prompt Enhancer applied successfully ({'I2I' if has_load_image else 'T2I'}, clip='{clip_model_name}', thinking=False).")
