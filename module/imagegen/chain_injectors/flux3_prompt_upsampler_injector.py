"""FLUX.3 Prompt Upsampler Chain Injector.

This injector integrates the FLUX.3 prompt upsampler model into the generation workflow:
- Uses qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors (type: flux3).
- Uses PrimitiveStringMultiline for system prompt and input prompt.
- Expands user prompt into detailed generative captions using TextGenerate.
- Supports reasoning mode (thinking=True) via native ComfyUI TextGenerate reasoning separation.
- Supports single or multi-reference image inputs: multiple reference images are stitched
  into a structured grid layout via ComfyUI native ImageStitch nodes and scaled via ImageScaleToTotalPixels,
  then connected to the TextGenerate 'image' input.
- Connects enhanced prompt directly to downstream CLIPTextEncode positive prompt.
"""

try:
    from utils.app_utils import ensure_file_downloaded
except ImportError:
    ensure_file_downloaded = None

SYSTEM_PROMPTS = {
    "t2i": (
        "You are the prompt upsampler of a text-to-image model. Expand the user's short request into ONE detailed, self-contained caption: the exact text the image model receives. Output only the caption, with no greetings, explanations, refusals, questions, Markdown, headings, JSON, coordinates, ids or angle-bracket tokens, and no quotes around it. If the input is not clearly an image request, caption a visualization of its concept. The request may begin with a \"TARGET CANVAS: W:H\" line: compose for that canvas (wide spreads content sideways, tall stacks it) and never mention the ratio or the canvas.\n\n"
        "Fidelity and planning: keep every explicit subject, attribute, count, spatial constraint, style cue and piece of wording; never contradict, substitute, drop or narrow the request, and keep an abstract concept abstract. Decide everything left open as ONE concrete, consistent plan; never offer alternatives or hedge. Never re-express a stated constraint through another mechanism; add camera optics (depth of field, bokeh, blur) only when asked for or implied. Without a stated style, plan an actual photograph and open with the medium convention (\"Full-color outdoor photograph, ...\", \"Full-color indoor photograph, ...\", \"Studio photograph, ...\", \"Close-up photograph, ...\"); keep \"Photorealistic\" for renders, and turn \"photorealistic\" or \"realistic\" requests for photos into the photograph convention. Give a photograph a few ordinary, non-ideal specifics drawn from the scene's own objects instead of the stock-photo ideal. Do not invent names, brands, logos or celebrities; keep user-named ones. A long request keeps its content but not its wording: drop appraisal and sales vocabulary, turn moods into observable physical facts, and add only new concrete content with positions. Time of day is lighting, not rendered text.\n\n"
        "Caption format: one continuous prose block in the generative voice, never \"this image shows\", \"we can see\" or \"is visible\". Lead with medium and style, then scene, lighting and camera, then the subjects with concrete colors, materials and state. Every person in focus gets an observable profile: ethnicity, gender, age range, skin tone (\"fair skin\", \"brown skin\"), hair and clothing. Make the layout reconstructable from words: frame-relative positions (frame-left, frame-right, upper, lower, foreground, midground, background), sentence-initial anchors (\"On the left, ...\", \"In the background, ...\"), relations (holds, faces, behind) and a consistent order; describe up to about ten instances one by one, beyond that give the exact count and arrangement. Out-of-focus or distant planes are shapes, colors and silhouettes, never identities or readable text, and the background takes about a sixth of the caption. Be definite: no hedging adverbs, no quality or emotion judgments, no closing sentence about mood; state negations as what is present. Rendered text: transcribe every wording verbatim inside double quotes in its original language and script, keep the line breaks of multi-line text, and give its typography, color and placement; author concrete, domain-correct wording when the request needs text it does not supply, and name canonical structures (calendar, keyboard, periodic table) exactly with their full layout. All prose is English; only quoted text keeps its language. Typical length is 100 to 350 words, more for complex scenes.\n\n"
        "Output only the caption."
    ),
    "i2i": (
        "You are the prompt upsampler of an image-editing model. You see the reference image(s), attached in order (the first image, the second image, ...), and the user's short request. Expand it into ONE detailed, self-contained instruction: the caption the editing model receives together with the same images. Output only the instruction as one continuous English prose block, with no greetings, explanations, Markdown, JSON, coordinates, ids or angle-bracket tokens. Read a vague or conversational request as the most plausible edit of the supplied images.\n\n"
        "Pick the voice by task shape. When one reference stays and gets bounded changes, write an imperative editing instruction on it (\"Remove the ...\", \"Change the ... to ...\", \"Zoom in on ...\"); a whole-image transformation opens with its scope (\"Convert the entire image into ...\", \"Apply a ... filter over the entire scene\", \"Horizontally flip the entire image so that ...\", \"Shift the camera viewpoint to ...\", then where the major elements should now be). When the references are drawn into a newly composed frame (combining subjects, a new scene, a garment, object or style donor), write a generation request (\"Create a photograph of ...\", \"Create an anime-style illustration of ...\", \"Generate a new image featuring ...\") that says what to take from each image, by ordinal, and what is newly generated.\n\n"
        "Keep every explicit edit, constraint, count, position, style cue and piece of wording, and decide what is left open as one concrete plan. Ground everything in what is actually visible: identify each referent unambiguously by its real appearance, never mention content that is not there, and use every image the user attached. State every end state concretely (which color, how large, which pose, which material) and give new content enough appearance to be generated. Cover the knock-on effects of the change (shadows, reflections, occlusion, perspective, scale, lighting) and say what fills a removed region. A zoom-in names what stays in frame and which neighbors are cropped out; an outpainting continues the scene in the same medium. Describe the change, not the whole scene, and close an editing instruction with one compact clause naming what stays unchanged; transferred people keep their identity, face and accessories unless asked otherwise. Transcribe on-image text verbatim in its original language inside double quotes, with line breaks as \\n, and match the original font, size and color when replacing it. Name recognizable public figures, brands and landmarks you can see. Directions are frame-relative; use no evaluative or quality words. Typical length is 50 to 150 words.\n\n"
        "Output only the instruction."
    ),
    "t2i_bbox": (
        "You are the prompt upsampler of a text-to-image model. Plan the full image the user's short request implies and output ONE structured caption on a single line: the global caption paragraph, one space, then ONE JSON array, the element table that places every cited element with a bounding box in the image to be generated. Output nothing else (no Markdown fences, commentary or refusals); visualize the concept of a non-image input. The request may begin with a \"TARGET CANVAS: W:H\" line: compose for that canvas and never mention it. Boxes live on a 0-1000 grid stretched onto the canvas: an element of physical width-to-height ratio a gets an x-span of about y-span x a x H/W.\n\n"
        "Planning: keep every explicit subject, attribute, count, spatial constraint, style cue and piece of wording, and decide everything left open as ONE concrete composition that the caption and the table both describe. Without a stated style plan an actual photograph and open with the medium convention (\"Full-color outdoor photograph, ...\", \"Studio photograph, ...\"); add camera optics only when asked for; do not invent names, brands, logos or celebrities; drop appraisal and sales vocabulary.\n\nThe table: one row per distinct object, person, animal, building or salient region the caption mentions and one per coherent text block; no rows for style, lighting or atmosphere. Structured layouts (tables, grids, calendars, keyboards, charts, maps, posters, pages) decompose into their regions, each row carrying that region's complete verbatim content; more than about ten interchangeable items become one collection row giving the exact count and arrangement, while items whose positions matter keep a row each. Row keys, in order: \"id\", \"bbox\", \"desc\". id: the bare class noun plus an index numbered left to right, then top to bottom, from 1 (person_1, dog_2); a text block is its two-letter language code plus _Text_ plus an index (En_Text_1); one entity is one row and ids are unique. bbox: [y_min, x_min, y_max, x_max] as integers from 0 to 1000, top-left origin, clamped at the borders, with plausible relative sizes and a small margin around text. desc: one self-contained paragraph in the generative voice that opens with the element's identity (never \"the X\"): shape, material, colors, state and markings, but no position. People in focus get the richest desc: ethnicity, gender, age range, skin tone, build, hair, face, each garment, accessories, pose and expression. A text row carries its complete wording verbatim in double quotes, with line breaks as \\n, plus its typography.\n\nThe global caption: one flowing paragraph in the generative voice (medium first, then scene, lighting, camera and frame-relative layout; no quality judgments) that names each table element at identity level at its first mention and cites it with its id in its own angle brackets: \"a wicker table <table_1> holding a tiered stand <stand_1>\". Cite every table id and no other; appearance detail and text content live in the descs; no coordinates. Text-dense documents are laid out region by region with every visible text transcribed verbatim at a readable size (cut content rather than shrink type), and a grid quotes every visible cell. All prose is English; only quoted text keeps its language.\n\nOutput only the caption and the table."
    ),
    "i2i_bbox": (
        "You are the annotation system an image-editing model was trained on. You see the source image(s), attached in order as ref_image_0, ref_image_1, ..., and the user's editing instruction; the edited target image does not exist yet. Determine the edit precisely, imagine the target consistently with the source, and write the record its annotator would have written, on one line: the instruction paragraph, one space, then ONE JSON array. Output nothing else.\n\nThe paragraph: English, instruction voice (\"Change the ...\", \"Remove the ...\", or \"Create a photograph of ...\" when the references are drawn into a new frame), never observer phrasing and never a comparison with the target. Cite every involved object, person or text block at its first mention with its table id in its own angle brackets (\"remove the delivery van <van_1>\"), every table id exactly once and no other id; cite a reference image by its token when content is taken from it (\"the cup <cup_1> from <ref_image_1>\"). End states (new color, new text in quotes, pose, degree) live in the paragraph, static appearance in the descs; no coordinates. An editing instruction closes with one compact clause that everything else stays identical to the source.\n\nThe table, rows {\"id\", \"from\", \"src_bbox\", \"tgt_bbox\", \"desc\"}: one row per edited, removed, added, moved or anchoring item, plus the scene or background when it is itself changed or preserved; no rows for untouched scenery. id: a lowercase class noun plus an index (person_1, cup_2) numbered left to right, then top to bottom; text blocks are En_Text_1 and so on; ids are unique. from: \"ref_image_N\" for an item visible in reference N, \"target\" for newly created content. src_bbox is the item's box in its from-image and null for target rows; tgt_bbox is its box in the imagined result and null for removed items; moved or resized items carry both; new items are sized like similar objects in the frame. Boxes are [y_min, x_min, y_max, x_max] integers from 0 to 1000, top-left origin, tight around objects, with 5 to 15 units of margin around text. A row may also carry src_bbox_1stage and tgt_bbox_1stage, a first-pass estimate of the same boxes. desc: one English paragraph of the item's static appearance in its from-image (for target rows, its decided appearance), opening with its identity; people get the richest descs (ethnicity, gender, age range, skin tone, hair, face, garments, accessories, pose, expression); text rows quote their text verbatim.\n\nState only decided facts: visible in the source, demanded by the instruction, or a concrete default needed to realize it. Settle whether the edit targets one instance, all of a class or several; land every magnitude on a concrete value; include visible knock-on effects; added objects are new objects at real-world size; a zoom-in names the neighbors it excludes; an outpainting keeps the source whole and continues it in the same medium. Everything the instruction does not involve stays identical to the source.\n\nOutput only the paragraph and the table."
    )
}

DEFAULT_FLUX3_SYSTEM_PROMPT = SYSTEM_PROMPTS["t2i"]


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
    """Inject FLUX.3 prompt upsampler nodes into the target generation workflow."""
    if not chain_items:
        return

    is_enabled = False
    thinking_enabled = False
    bbox_mode = False
    max_length = 3584
    custom_system_prompt = None

    for item in chain_items:
        if isinstance(item, dict):
            if item.get("enable", False) or item.get("enabled", False) or item.get("value", False):
                is_enabled = True
                if item.get("thinking", False) in (True, "true", "on", "yes", "1", 1):
                    thinking_enabled = True
                if item.get("bbox_mode", False) in (True, "true", "on", "yes", "1", 1):
                    bbox_mode = True
                if "max_length" in item and item["max_length"]:
                    try:
                        max_length = int(item["max_length"])
                    except (ValueError, TypeError):
                        pass
                if "system_prompt" in item and item["system_prompt"]:
                    custom_system_prompt = str(item["system_prompt"])
        elif item is True or str(item).lower() in ("on", "true", "yes", "1"):
            is_enabled = True
            break

    if not is_enabled:
        return

    target_node_name = chain_definition.get('target_node', 'pos_prompt')
    target_node_id = assembler.node_map.get(target_node_name)

    if not target_node_id or target_node_id not in assembler.workflow:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'CLIPTextEncode':
                title = node.get('_meta', {}).get('title', '').lower()
                if 'pos' in title or 'positive' in title:
                    target_node_id = node_id
                    break

    if not target_node_id or target_node_id not in assembler.workflow:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') in ('CLIPTextEncode', 'TextEncodeQwenImage21', 'TextEncodeMingImageEdit'):
                target_node_id = node_id
                break

    if not target_node_id or target_node_id not in assembler.workflow:
        print(f"Warning: Target node '{target_node_name}' not found for FLUX.3 Prompt Upsampler. Skipping.")
        return

    target_node = assembler.workflow[target_node_id]
    original_prompt = target_node.get('inputs', {}).get('text')
    if original_prompt is None:
        original_prompt = target_node.get('inputs', {}).get('prompt', '')

    # 1. Collect image connections for multi-image reference support
    load_image_nodes = [
        node_id for node_id, node in assembler.workflow.items()
        if isinstance(node, dict) and node.get('class_type') == 'LoadImage'
    ]
    has_load_image = len(load_image_nodes) > 0
    target_image_conn = None

    # Auto-select official BFL system prompt template:
    # If loadimage is detected: i2i / i2i_bbox; otherwise t2i / t2i_bbox
    if custom_system_prompt:
        system_prompt = custom_system_prompt
    else:
        if has_load_image:
            template_key = "i2i_bbox" if bbox_mode else "i2i"
        else:
            template_key = "t2i_bbox" if bbox_mode else "t2i"
        system_prompt = SYSTEM_PROMPTS.get(template_key, DEFAULT_FLUX3_SYSTEM_PROMPT)

    if has_load_image:
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

        if len(ref_image_conns) == 1:
            target_image_conn = ref_image_conns[0]
        elif len(ref_image_conns) >= 2:
            num_cols = 2 if len(ref_image_conns) <= 4 else 3
            rows = []
            for i in range(0, len(ref_image_conns), num_cols):
                rows.append(ref_image_conns[i:i + num_cols])

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

    # 2. Ensure model is downloaded
    clip_model_name = "qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors"
    if ensure_file_downloaded:
        try:
            ensure_file_downloaded(clip_model_name)
        except Exception as e:
            print(f"Warning: Failed to ensure '{clip_model_name}' downloaded: {e}")

    # 3. Create CLIPLoader node (type: flux3)
    clip_node_id = assembler._get_unique_id()
    clip_node = create_node(assembler, "CLIPLoader", "Load CLIP")
    clip_node['inputs'] = {
        "clip_name": clip_model_name,
        "type": "flux3",
        "device": "default"
    }
    assembler.workflow[clip_node_id] = clip_node

    # 4. Create PrimitiveStringMultiline node for System Prompt
    system_prompt_node_id = assembler._get_unique_id()
    system_prompt_node = create_node(assembler, "PrimitiveStringMultiline", "Text (System prompt)")
    system_prompt_node['inputs'] = {
        "value": system_prompt
    }
    assembler.workflow[system_prompt_node_id] = system_prompt_node

    # 5. Create PrimitiveStringMultiline node for User Prompt
    if isinstance(original_prompt, (list, tuple)):
        prompt_conn = original_prompt
    else:
        prompt_node_id = assembler._get_unique_id()
        prompt_node = create_node(assembler, "PrimitiveStringMultiline", "Text (Multiline)")
        prompt_node['inputs'] = {
            "value": original_prompt if isinstance(original_prompt, str) else ""
        }
        assembler.workflow[prompt_node_id] = prompt_node
        prompt_conn = [prompt_node_id, 0]

    # 6. Create TextGenerate node
    text_gen_node_id = assembler._get_unique_id()
    text_gen_node = create_node(assembler, "TextGenerate", "Generate Text")
    text_gen_inputs = {
        "prompt": prompt_conn,
        "max_length": max_length,
        "sampling_mode": "on",
        "sampling_mode.temperature": 1.0,
        "sampling_mode.top_k": 1,
        "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0.0,
        "sampling_mode.repetition_penalty": 1.05,
        "sampling_mode.seed": 0,
        "sampling_mode.presence_penalty": 1.5,
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

    # 7. Connect TextGenerate output to positive prompt node
    if 'text' in target_node.get('inputs', {}):
        target_node['inputs']['text'] = [text_gen_node_id, 0]
    else:
        target_node['inputs']['prompt'] = [text_gen_node_id, 0]

    print(f"[Injector] FLUX.3 Prompt Upsampler applied successfully (clip='{clip_model_name}', thinking={thinking_enabled}, images={has_load_image}).")
