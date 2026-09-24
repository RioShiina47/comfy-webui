"""Ming-Image Prompt Enhancer Chain Injector.

This injector automatically integrates the Qwen3.8-27B prompt enhancement model into
the Ming-Image generation workflow:
- Detects whether the task is pure text-to-image (T2I) or involves images (I2I, inpainting,
  outpainting, hires.fix, or multi-reference images).
- Uses the Qwen3.8-27B PE text encoder: qwen3.8_27b_w4a8.safetensors (type: qwen_image).
- Handles single or multi-reference inputs: multiple reference images are stitched into
  a structured grid layout via ComfyUI native ImageStitch nodes and scaled via ImageScaleToTotalPixels.
- Supports reasoning mode (thinking=True) leveraging ComfyUI TextGenerate native reasoning separation
  (output slot 0 outputs clean prompt text, while output slot 1 isolates reasoning chain).
"""

try:
    from utils.app_utils import ensure_file_downloaded
except ImportError:
    ensure_file_downloaded = None


DEFAULT_MING_IMAGE_SYSTEM_PROMPT = (
    "You are a senior visual designer and image-prompt engineer. Expand the user's request into one precise, high-resolution Figma-style caption. Return only one JSON object. "
    "Use exactly two top-level keys. `canvas_settings` contains exactly `aspect_ratio`, `ambient_lighting`, and `image_style`. `layers` lists visible groups from background to topmost overlay. Every layer contains exactly `description`, `coordinates`, `hierarchy_and_relation`, and `color_specs`; `color_specs` is an array of hex colors. "
    '`coordinates` MUST be one string, never an object or array, in exactly this form: `"cx: 0.500, cy: 0.500, w: 1.000, h: 1.000"`. Values are normalized; each bbox encloses its complete owned object and stays inside the canvas. '
    "A layer is one selectable visible semantic group: background, full person, coherent object, panel, card, row, or text block. Prefer the fewest groups that preserve the layout. Keep people and objects intact. Never create invisible parents, guides, placeholders, empty layers, duplicate summaries, or multiple owners for one element. "
    "Preserve every user-supplied rendered string character-for-character and as one contiguous string. Unless multiple visible copies are requested, it must occur exactly once across all `description` fields and zero times in `hierarchy_and_relation`. Quote it only where describing its visible rendering; refer to the related subject elsewhere with unquoted semantic wording. Enumerate intended copy, invent extra copy sparingly, and never hide content behind \"other text\", \"remaining labels\", or \"etc.\" "
    "Describe concrete composition, typography, materials, texture, lighting, pose, and camera treatment without literary filler. Use `hierarchy_and_relation` only for ownership, alignment, containment, stacking, and occlusion. "
    "Infer structured layouts first. Use one complete layer per card and state its row and column. A compact secondary table may be one layer only if every header and cell is listed; otherwise use a visible shared frame when present, one complete header, and one complete layer per body row, binding values to columns and stating blanks. Enumerate sequences, schedules, spans, gaps, and vacant tracks in visual order. Do not mistake ordinary alignment for a table. "
    "Silently verify schema, string coordinates, Z-order, exact-text counts, geometry, bbox validity, and completeness."
)


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
    """Inject Ming-Image prompt enhancer nodes into the target generation workflow."""
    if not chain_items:
        return

    is_enabled = False
    thinking_enabled = False
    max_length = 4096
    system_prompt = DEFAULT_MING_IMAGE_SYSTEM_PROMPT
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
                if "system_prompt" in item and item["system_prompt"]:
                    system_prompt = str(item["system_prompt"])
        elif item is True or str(item).lower() in ("on", "true", "yes", "1"):
            is_enabled = True
            break

    if not is_enabled:
        return

    target_node_name = chain_definition.get('target_node', 'prompt')
    target_node_id = assembler.node_map.get(target_node_name)

    if not target_node_id or target_node_id not in assembler.workflow:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'TextEncodeMingImageEdit':
                target_node_id = node_id
                break

    if not target_node_id or target_node_id not in assembler.workflow:
        print(f"Warning: Target node '{target_node_name}' (TextEncodeMingImageEdit) not found for Ming-Image Prompt Enhancer. Skipping.")
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
    clip_model_name = "qwen3.8_27b_w4a8.safetensors"

    if has_load_image:
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

    # 3. Create CLIPLoader node (using qwen_image type for Qwen3.8-27B PE)
    clip_node_id = assembler._get_unique_id()
    clip_node = create_node(assembler, "CLIPLoader", "Load CLIP")
    clip_node['inputs'] = {
        "clip_name": clip_model_name,
        "type": "qwen_image",
        "device": "default"
    }
    assembler.workflow[clip_node_id] = clip_node

    # 4. Create PrimitiveString node for System Prompt (Text node)
    system_prompt_node_id = assembler._get_unique_id()
    system_prompt_node = create_node(assembler, "PrimitiveString", "Text")
    system_prompt_node['inputs'] = {
        "value": system_prompt
    }
    assembler.workflow[system_prompt_node_id] = system_prompt_node

    # 5. Create TextGenerate node
    text_gen_node_id = assembler._get_unique_id()
    text_gen_node = create_node(assembler, "TextGenerate", "Generate Text")
    text_gen_inputs = {
        "prompt": original_prompt,
        "max_length": max_length,
        "sampling_mode": "on",
        "sampling_mode.temperature": 0.7,
        "sampling_mode.top_k": 64,
        "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0.05,
        "sampling_mode.repetition_penalty": 1.05,
        "sampling_mode.seed": 0,
        "sampling_mode.presence_penalty": 0,
        "thinking": thinking_enabled,
        "use_default_template": True,
        "mtp": "auto",
        "clip": [clip_node_id, 0],
        "system_prompt": [system_prompt_node_id, 0]
    }
    if target_image_conn is not None:
        text_gen_inputs["image"] = target_image_conn

    text_gen_node['inputs'] = text_gen_inputs
    assembler.workflow[text_gen_node_id] = text_gen_node

    # 6. Connect TextGenerate output directly to target node prompt
    # ComfyUI TextGenerate node natively strips reasoning tags (<think>...</think>) into output slot 1,
    # leaving clean enhanced prompt in output slot 0.
    target_node['inputs']['prompt'] = [text_gen_node_id, 0]
    print(f"[Injector] Ming-Image Prompt Enhancer applied successfully ({'I2I' if has_load_image else 'T2I'}, clip='{clip_model_name}', thinking={thinking_enabled}).")
