"""Prompt Enhancer Chain Injector (data-driven).

Merges the former qwen_image_2_1_prompt_enhancer / ming_image_prompt_enhancer /
flux3_prompt_upsampler injectors into a single implementation driven by
``yaml/prompt_enhancer.yaml``.

The YAML declares three independent modules, named after their UI labels in the
"Prompt Enhancer Settings" accordion:

* ``architecture`` Model framework -> default ``clip`` (Prompt Enhancer Model),
  optional ``clip_type`` override and default ``system_prompt`` (System Prompt).
  These are the dropdown defaults applied whenever the base model changes. A
  missing entry falls back to the first registry item.
* ``clip``         Every name shown in the "Prompt Enhancer Model" dropdown,
  mapped to its checkpoint ``filename`` and, when the checkpoint is only valid as
  one encoder, an optional ``clip_type``. Declaration order is dropdown order.
* ``system_prompt`` Every name shown in the "System Prompt" dropdown, mapped to
  the system prompt sent to ``TextGenerateV2``. An empty value means no system
  prompt.

The modules are fully decoupled: any framework may pick any clip, and any clip
may be paired with any system prompt. ``clip_type`` describes the *framework's*
encoder, not the checkpoint: the same Qwen3.5-9B weights are a ``minimax``
encoder for MiniMax-H3-Image and a ``moss_image`` encoder for MOSS-Image, and
the official ComfyUI workflows load them exactly that way. It is therefore
declared on every ``architecture`` entry and applies to whatever clip the user
selects, so a MOSS-Image run can host the Qwen-Image-2.1 upsampler as
``moss_image``. There is no image-based or bbox auto-detection.

The injector then:

1. Resolves the target text-encode node that receives the enhanced prompt.
2. Collects reference images (native ``images.image_N`` slots first, otherwise
   ``LoadImage`` nodes) and feeds each one into its own autogrow slot.
3. Loads the selected checkpoint through ``CLIPLoader`` and, when the chosen
   system prompt carries text, materialises it as a primitive string node.
4. Emits a ``TextGenerateV2`` node (Sampling Mode ``off``) and wires its clean
   ``generated_text`` output into the downstream positive prompt.
"""

import os

import yaml

try:
    from utils.app_utils import ensure_file_downloaded
except ImportError:
    ensure_file_downloaded = None


CHAIN_TYPE = "prompt_enhancer"

_YAML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yaml")
_CONFIG_PATH = os.path.join(_YAML_DIR, "prompt_enhancer.yaml")

# Node class -> (target node names to try, input keys holding the source prompt)
_TARGET_HINTS = (
    ("TextEncodeQwenImage21", ("prompt",), ("prompt",)),
    ("TextEncodeMingImageEdit", ("prompt",), ("prompt",)),
    ("PH3ImageConditioning", ("prompt",), ("prompt",)),
    ("MiniMaxH3ImageConditioning", ("prompt",), ("prompt",)),
    ("TextEncodeMOSSImageEdit", ("prompt",), ("prompt",)),
    ("Krea2EditGroundedEncode", ("prompt",), ("prompt",)),
    ("TextEncodeQwenImageEditPlus", ("prompt",), ("prompt",)),
    ("CLIPTextEncode", ("pos_prompt",), ("text", "prompt")),
)

_config_cache = None


def load_prompt_enhancer_config():
    """Load and cache the prompt enhancer registry."""
    global _config_cache
    if _config_cache is None:
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                _config_cache = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            print(f"Warning: Failed to load prompt enhancer config '{_CONFIG_PATH}': {e}")
            _config_cache = {}
    return _config_cache


def get_clip_registry():
    """Return the ordered ``clip name -> {filename[, clip_type]}`` registry."""
    return load_prompt_enhancer_config().get("clip", {}) or {}


def get_system_prompt_registry():
    """Return the ``system_prompt name -> text`` map (``""`` = no prompt)."""
    return load_prompt_enhancer_config().get("system_prompt", {}) or {}


def get_architectures():
    """Return the ``model framework -> {clip, system_prompt}`` defaults map."""
    return load_prompt_enhancer_config().get("architecture", {}) or {}


def get_clip_names():
    """Return the selectable clip names, in declared order."""
    return [str(name) for name in get_clip_registry().keys()]


def get_clip_entry(name):
    """Return the ``{filename[, clip_type]}`` registry entry for a clip name."""
    entry = get_clip_registry().get(str(name))
    return entry if isinstance(entry, dict) else {}


def get_system_prompt_names():
    """Return the selectable system_prompt names, including empty (no-prompt) ones."""
    return [str(name) for name in get_system_prompt_registry().keys()]


def get_all_system_prompt_keys():
    """Return the system_prompt names that carry text (empty ones are dropped)."""
    return [name for name, text in get_system_prompt_registry().items() if text]


def get_architecture_defaults(framework=None):
    """Return ``{model, system_prompt, clip_type}`` defaults for a model framework.

    Reads the ``architecture`` module. Falls back to the first registry entry
    when the framework is unknown or omits a field, so the UI always has a
    usable default.
    """
    defaults = get_architectures().get(str(framework), {}) if framework else {}
    if not isinstance(defaults, dict):
        defaults = {}

    clip_names = get_clip_names()
    prompt_names = get_system_prompt_names()

    model = defaults.get("clip")
    if model not in clip_names:
        model = clip_names[0] if clip_names else None

    system_prompt = defaults.get("system_prompt")
    if system_prompt not in prompt_names:
        system_prompt = prompt_names[0] if prompt_names else None

    return {
        "model": model,
        "system_prompt": system_prompt,
        "clip_type": defaults.get("clip_type"),
    }


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


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("on", "true", "yes", "1")


def _parse_chain_items(chain_items):
    """Normalise the chain items into ``{enable, thinking, model, system_prompt, max_length}``."""
    params = {
        "enable": False,
        "thinking": False,
        "model": None,
        "system_prompt": None,
        "max_length": None,
    }
    if isinstance(chain_items, dict):
        chain_items = [chain_items]

    for item in chain_items or []:
        if isinstance(item, dict):
            if item.get("enable", False) or item.get("enabled", False) or item.get("value", False):
                params["enable"] = True
                params["thinking"] = _as_bool(item.get("thinking"), False)
                if item.get("model"):
                    params["model"] = str(item["model"])
                if item.get("system_prompt"):
                    params["system_prompt"] = str(item["system_prompt"])
                if item.get("max_length"):
                    try:
                        params["max_length"] = int(item["max_length"])
                    except (ValueError, TypeError):
                        pass
        elif item is True or str(item).strip().lower() in ("on", "true", "yes", "1"):
            params["enable"] = True
            break

    return params


def _infer_clip_type(clip_filename):
    """Best-effort ``CLIPLoader`` type when ``clip_type`` is missing from the YAML.

    A wrong type silently loads the checkpoint into the wrong text encoder, so the
    filename is inspected before falling back to the most common type.
    """
    name = str(clip_filename or "").lower()
    if "flux3" in name:
        print(f"Warning: 'clip_type' missing for '{clip_filename}'; inferred 'flux3' from the filename.")
        return "flux3"
    if "flux2" in name:
        print(f"Warning: 'clip_type' missing for '{clip_filename}'; inferred 'flux2' from the filename.")
        return "flux2"
    if "p-h3" in name or "minimax" in name:
        print(f"Warning: 'clip_type' missing for '{clip_filename}'; inferred 'minimax' from the filename.")
        return "minimax"
    if "moss" in name:
        print(f"Warning: 'clip_type' missing for '{clip_filename}'; inferred 'moss_image' from the filename.")
        return "moss_image"
    if "krea2" in name or "krea-2" in name:
        print(f"Warning: 'clip_type' missing for '{clip_filename}'; inferred 'krea2' from the filename.")
        return "krea2"
    print(f"Warning: 'clip_type' missing for '{clip_filename}'; defaulting to 'qwen_image'. Add it to prompt_enhancer.yaml.")
    return "qwen_image"


def _resolve_clip_type(clip_name, framework=None):
    """Resolve the ``CLIPLoader.type`` to load the Prompt Enhancer checkpoint with.

    ``clip_type`` is a property of the *framework*, not of the checkpoint: the
    same Qwen3-VL-4B weights are loaded as ``flux3`` by FLUX.3, ``krea2`` by
    Krea-2, ``mage`` by Mage-Flow and ``qwen_image`` by Qwen-Image, and the same
    Qwen3.5-9B weights as ``minimax`` by MiniMax-H3-Image and ``moss_image`` by
    MOSS-Image. The official ComfyUI workflows do exactly this.

    A prompt enhancer is prompt engineering on top of an LLM, so any registered
    upsampler may run under any framework as long as the framework's encoder
    class can host those weights. Resolution order:

    1. ``architecture[framework].clip_type`` - always wins, including for a clip
       the user picked from another architecture (MOSS-Image + Qwen-Image-2.1 PE
       still loads as ``moss_image``).
    2. ``clip[name].clip_type`` - only for a checkpoint that genuinely cannot be
       re-hosted, and for calls with no framework context.
    3. Filename inference, then ``qwen_image``.
    """
    defaults = get_architecture_defaults(framework)
    entry = get_clip_entry(clip_name)
    if defaults.get("clip_type"):
        return str(defaults["clip_type"])
    if entry.get("clip_type"):
        return str(entry["clip_type"])
    return _infer_clip_type(str(entry.get("filename") or clip_name))


def _resolve_model(params, chain_definition=None):
    """Pick the Prompt Enhancer Model: explicit request, then framework default, then first registered."""
    available = get_clip_names()
    if not available:
        return None

    requested = params.get("model")
    if not requested:
        defaults = get_architecture_defaults((chain_definition or {}).get("framework"))
        requested = defaults["model"]

    if requested:
        requested = str(requested)
        if requested in available:
            return requested
        # Tolerate the checkpoint filename being supplied instead of the name.
        # One checkpoint can be registered twice with different clip_types (the
        # MiniMax-H3 PE upsampler is a `minimax` encoder, the same file is a
        # `qwen_image` encoder for other frameworks), so prefer the framework's
        # own default before falling back to declaration order.
        by_filename = [
            name for name in available
            if str(get_clip_entry(name).get("filename") or "") == requested
        ]
        if by_filename:
            framework_default = get_architecture_defaults(
                (chain_definition or {}).get("framework")
            )["model"]
            return framework_default if framework_default in by_filename else by_filename[0]
        # Unknown name: fall through to the framework default.

    defaults = get_architecture_defaults((chain_definition or {}).get("framework"))
    return defaults["model"] if defaults["model"] in available else available[0]


def _select_system_prompt(requested_system_prompt, framework=None):
    """Resolve the system prompt text from the global system_prompt registry."""
    prompts = get_system_prompt_registry()
    if not prompts:
        return None

    # An explicit UI choice always wins, including an empty prompt (no prompt).
    if requested_system_prompt and str(requested_system_prompt) in prompts:
        return prompts[str(requested_system_prompt)] or None

    # Otherwise use the framework default, falling back to any non-empty prompt.
    default_prompt = get_architecture_defaults(framework)["system_prompt"]
    if default_prompt and str(default_prompt) in prompts:
        return prompts[str(default_prompt)] or None

    for text in prompts.values():
        if text:
            return text
    return None


def _find_target_node(assembler, chain_definition=None):
    """Locate the text-encode node that should receive the enhanced prompt."""
    target_node_name = (chain_definition or {}).get("target_node")
    target_node_id = assembler.node_map.get(target_node_name) if target_node_name else None
    if target_node_id and target_node_id in assembler.workflow:
        return target_node_id, None

    for class_type, node_names, input_keys in _TARGET_HINTS:
        return_class = (class_type, input_keys)
        for node_name in node_names:
            node_id = assembler.node_map.get(node_name)
            if node_id and assembler.workflow.get(node_id, {}).get('class_type') == class_type:
                return node_id, return_class

        # Prefer a positive-prompt encode node when the class_type is shared.
        for node_id, node in assembler.workflow.items():
            if not isinstance(node, dict) or node.get('class_type') != class_type:
                continue
            title = str(node.get('_meta', {}).get('title', '')).lower()
            if 'pos' in title or 'positive' in title:
                return node_id, return_class

        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == class_type:
                return node_id, return_class

    return None, None


def _get_autogrow_slot_names(class_type, input_name='images'):
    """Resolve the per-image autogrow slot names for ``class_type.input_name``.

    ComfyUI exposes two flavours of Autogrow template:

    * ``TemplateNames(names=["image1", ...])`` -> explicit names
    * ``TemplatePrefix(prefix="image", max=N)`` -> ``prefix + range(max)``

    Returns ``None`` when the node schema cannot be introspected.
    """
    try:
        from comfy_integration.nodes import NODE_CLASS_MAPPINGS
        node_class = NODE_CLASS_MAPPINGS.get(class_type)
    except Exception:
        node_class = None

    if node_class is not None:
        try:
            input_types = node_class.INPUT_TYPES() or {}
        except Exception:
            input_types = {}
        for category in ('required', 'optional'):
            spec = (input_types.get(category) or {}).get(input_name)
            if not spec or len(spec) < 2 or not isinstance(spec[1], dict):
                continue
            template = spec[1].get('template')
            if not isinstance(template, dict):
                continue
            names = template.get('names')
            if isinstance(names, (list, tuple)) and names:
                return [str(n) for n in names]
            prefix = template.get('prefix')
            if isinstance(prefix, str):
                max_count = template.get('max', 100)
                if not isinstance(max_count, int) or max_count <= 0:
                    max_count = 100
                return [f"{prefix}{i}" for i in range(max_count)]

    # Fall back to the published TextGenerateV2 contract.
    if class_type == "TextGenerateV2":
        return [f"image{i}" for i in range(1, 51)]
    return None


def _collect_reference_images(assembler, target_node):
    """Collect the reference image connections feeding the prompt node.

    Native autogrow slots on the target node (``images.image_N``) win; otherwise
    every ``LoadImage`` node is scaled and used. Each image keeps its own
    connection so the caller can wire them into separate autogrow slots.
    """
    load_image_nodes = [
        node_id for node_id, node in assembler.workflow.items()
        if isinstance(node, dict) and node.get('class_type') == 'LoadImage'
    ]

    indexed_slots = []
    for slot_key, slot_val in target_node.get('inputs', {}).items():
        if slot_key.startswith('images.image_') and isinstance(slot_val, (list, tuple)):
            try:
                idx = int(slot_key.split('_')[-1])
            except (ValueError, IndexError):
                idx = 999
            indexed_slots.append((idx, slot_val))

    indexed_slots.sort(key=lambda x: x[0])
    if indexed_slots:
        return [slot[1] for slot in indexed_slots]

    if not load_image_nodes:
        return []

    scaled_conns = []
    for load_id in load_image_nodes:
        scaled_conn = None
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'ImageScaleToTotalPixels':
                img_input = node.get('inputs', {}).get('image')
                if isinstance(img_input, (list, tuple)) and str(img_input[0]) == str(load_id):
                    scaled_conn = [node_id, 0]
                    break
        if scaled_conn is None:
            scale_id = assembler._get_unique_id()
            scale_node = create_node(assembler, "ImageScaleToTotalPixels", "Scale Input Image")
            scale_node['inputs'] = {
                "upscale_method": "nearest-exact",
                "megapixels": 1,
                "resolution_steps": 1,
                "image": [load_id, 0]
            }
            assembler.workflow[scale_id] = scale_node
            scaled_conn = [scale_id, 0]
        scaled_conns.append(scaled_conn)
    return scaled_conns


def inject(assembler, chain_definition, chain_items):
    """Inject prompt enhancer nodes into the target generation workflow."""
    if not chain_items:
        return

    params = _parse_chain_items(chain_items)
    if not params["enable"]:
        return

    model_name = _resolve_model(params, chain_definition)
    if not model_name:
        print("Warning: No Prompt Enhancer Models configured in prompt_enhancer.yaml. Skipping.")
        return

    framework = (chain_definition or {}).get("framework")
    entry = get_clip_entry(model_name)
    clip_model_name = str(entry.get("filename") or model_name)
    clip_type = _resolve_clip_type(model_name, framework)

    target_node_id, class_hint = _find_target_node(assembler, chain_definition)
    if not target_node_id:
        print(f"Warning: Target text-encode node for prompt enhancer '{model_name}' not found. Skipping.")
        return

    target_node = assembler.workflow[target_node_id]

    # Fall back to the detected node class when the recipe gave no target_node.
    prompt_input_keys = (class_hint[1] if class_hint else None) or ("text", "prompt")
    original_prompt = None
    for input_key in prompt_input_keys:
        original_prompt = target_node.get('inputs', {}).get(input_key)
        if original_prompt is not None:
            break
    if original_prompt is None:
        original_prompt = ""

    image_conns = _collect_reference_images(assembler, target_node)
    system_prompt_name = params["system_prompt"] if params["system_prompt"] else None
    if ensure_file_downloaded:
        try:
            ensure_file_downloaded(clip_model_name)
        except Exception as e:
            print(f"Warning: Failed to ensure '{clip_model_name}' downloaded: {e}")

    # 1. Load the selected Prompt Enhancer Model checkpoint. Reuse an existing
    #    CLIPLoader with the same checkpoint and type: MOSS-Image upsamples with the
    #    very encoder it conditions on, and a second loader would hold the same
    #    7 GB checkpoint twice.
    clip_node_id = None
    for existing_id, existing_node in assembler.workflow.items():
        if not isinstance(existing_node, dict) or existing_node.get('class_type') != 'CLIPLoader':
            continue
        existing_inputs = existing_node.get('inputs', {})
        if (existing_inputs.get('clip_name') == clip_model_name
                and existing_inputs.get('type') == clip_type):
            clip_node_id = existing_id
            print(f"[Injector] Reusing existing CLIPLoader '{existing_id}' for prompt enhancer.")
            break

    if clip_node_id is None:
        clip_node_id = assembler._get_unique_id()
        clip_node = create_node(assembler, "CLIPLoader", "Load CLIP")
        clip_node['inputs'] = {
            "clip_name": clip_model_name,
            "type": clip_type,
            "device": "default"
        }
        assembler.workflow[clip_node_id] = clip_node

    # 2. Materialise the system prompt when the selected System Prompt carries text
    system_prompt_conn = None
    system_prompt = _select_system_prompt(system_prompt_name, framework)
    if system_prompt:
        system_prompt_node_id = assembler._get_unique_id()
        system_prompt_node = create_node(assembler, "PrimitiveStringMultiline", "Text (System prompt)")
        system_prompt_node['inputs'] = {"value": system_prompt}
        assembler.workflow[system_prompt_node_id] = system_prompt_node
        system_prompt_conn = [system_prompt_node_id, 0]

    # 3. Emit the TextGenerateV2 node (Sampling Mode off: sampling-only sub-inputs are omitted)
    max_length = params["max_length"] or 4096
    text_gen_node_id = assembler._get_unique_id()
    text_gen_node = create_node(assembler, "TextGenerateV2", "Generate Text")
    text_gen_inputs = {
        "prompt": original_prompt,
        "max_length": max_length,
        "sampling_mode": "off",
        "thinking": params["thinking"],
        "use_default_template": True,
        "mtp": "auto",
        "clip": [clip_node_id, 0]
    }
    if system_prompt_conn is not None:
        text_gen_inputs["system_prompt"] = system_prompt_conn

    # Wire each reference image into its own autogrow slot (image1, image2, ...).
    if image_conns:
        slot_names = _get_autogrow_slot_names("TextGenerateV2", "images")
        if slot_names:
            for conn, slot_name in zip(image_conns, slot_names):
                text_gen_inputs[slot_name] = conn
            if len(image_conns) > len(slot_names):
                print(
                    f"Warning: {len(image_conns)} reference images but TextGenerateV2 "
                    f"exposes only {len(slot_names)} slots; extra images were dropped."
                )
        else:
            print("Warning: could not resolve TextGenerateV2 autogrow slots; images not wired.")

    text_gen_node['inputs'] = text_gen_inputs
    assembler.workflow[text_gen_node_id] = text_gen_node

    # 4. Route the clean generated_text output into the downstream prompt input
    # (slot 0 carries the enhanced prompt; ComfyUI puts reasoning into slot 1)
    target_inputs = target_node.setdefault('inputs', {})
    if 'text' in target_inputs or ('prompt' not in target_inputs and 'text' in prompt_input_keys):
        target_inputs['text'] = [text_gen_node_id, 0]
    else:
        target_inputs['prompt'] = [text_gen_node_id, 0]

    print(
        f"[Injector] Prompt Enhancer '{model_name}' applied successfully "
        f"(clip='{clip_model_name}', type='{clip_type}', "
        f"system_prompt={system_prompt_name or 'auto'})."
    )
