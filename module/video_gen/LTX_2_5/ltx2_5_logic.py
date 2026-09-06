import os
from core.workflow_assembler import WorkflowAssembler
from core.utils import get_filename_prefix, handle_seed, save_temp_image, save_temp_audio
from core.input_processors import process_lora_inputs

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPES_DIR = os.path.join(MODULE_DIR, "workflow_recipes")

RESOLUTION_PRESETS = {
    "1080p": {
        "16:9 (Widescreen)": (1920, 1088),
        "9:16 (Vertical)": (1088, 1920),
        "1:1 (Square)": (1440, 1440),
        "4:3 (Classic TV)": (1664, 1248),
        "3:4 (Classic Portrait)": (1248, 1664),
        "3:2 (Photography)": (1760, 1184),
        "2:3 (Photography Portrait)": (1184, 1760),
    },
    "768p": {
        "16:9 (Widescreen)": (1344, 768),
        "9:16 (Vertical)": (768, 1344),
        "1:1 (Square)": (1024, 1024),
        "4:3 (Classic TV)": (1152, 864),
        "3:4 (Classic Portrait)": (864, 1152),
        "3:2 (Photography)": (1248, 832),
        "2:3 (Photography Portrait)": (832, 1248),
    },
    "544p": {
        "16:9 (Widescreen)": (960, 544),
        "9:16 (Vertical)": (544, 960),
        "1:1 (Square)": (736, 736),
        "4:3 (Classic TV)": (832, 640),
        "3:4 (Classic Portrait)": (640, 832),
        "3:2 (Photography)": (864, 576),
        "2:3 (Photography Portrait)": (576, 864),
    }
}

VALID_TASKS = {"T2VA", "I2VA", "TA2VA", "IA2VA", "FLF2VA"}


def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    task = str(local_ui_values.get('task', 'T2VA')).upper()
    if task not in VALID_TASKS:
        task = "T2VA"

    task_prefix = f"ltx2_5_{task.lower()}"

    # 1. Select Recipe based on upscalers
    use_spatial = local_ui_values.get('use_spatial_upscaler', False)
    use_temporal = local_ui_values.get('use_temporal_upscaler', False)

    if use_spatial and use_temporal:
        recipe_filename = f"{task_prefix}_3x_recipe.yaml"
    elif use_spatial:
        recipe_filename = f"{task_prefix}_2x_recipe.yaml"
        local_ui_values['upscaler_model_name'] = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    elif use_temporal:
        recipe_filename = f"{task_prefix}_2x_recipe.yaml"
        local_ui_values['upscaler_model_name'] = "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
    else:
        recipe_filename = f"{task_prefix}_recipe.yaml"

    recipe_path = os.path.join(RECIPES_DIR, recipe_filename)

    # 2. Resolution & Dimensions
    resolution = str(local_ui_values.get('resolution', '768p')).lower()
    preset_dict = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS["768p"])
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Widescreen)")
    width, height = preset_dict.get(selected_ratio, (1344, 768))
    local_ui_values['width'] = width
    local_ui_values['height'] = height

    # 3. Seed
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    # 4. FPS
    fps_raw = local_ui_values.get('fps', "24fps")
    if isinstance(fps_raw, str):
        fps = int(fps_raw.replace('fps', '').replace('FPS', '').strip())
    else:
        fps = int(fps_raw)
    local_ui_values['frame_rate'] = fps

    # 5. Prompts & Task-specific media handling
    if task in ("I2VA", "TA2VA"):
        pos_prompt = local_ui_values.get('side_positive_prompt') or local_ui_values.get('full_positive_prompt') or local_ui_values.get('positive_prompt') or ""
        neg_prompt = local_ui_values.get('side_negative_prompt') or local_ui_values.get('full_negative_prompt') or local_ui_values.get('negative_prompt') or "pc game, console game, video game, cartoon, childish, ugly"
    else:
        pos_prompt = local_ui_values.get('full_positive_prompt') or local_ui_values.get('side_positive_prompt') or local_ui_values.get('positive_prompt') or ""
        neg_prompt = local_ui_values.get('full_negative_prompt') or local_ui_values.get('side_negative_prompt') or local_ui_values.get('negative_prompt') or "pc game, console game, video game, cartoon, childish, ugly"
    local_ui_values['positive_prompt'] = pos_prompt
    local_ui_values['negative_prompt'] = neg_prompt

    start_img = local_ui_values.get('i2va_start_image') or local_ui_values.get('stacked_start_image') or local_ui_values.get('start_image')
    end_img = local_ui_values.get('stacked_end_image') or local_ui_values.get('end_image')
    audio_file = local_ui_values.get('ta2va_audio_file') or local_ui_values.get('stacked_audio_file') or local_ui_values.get('audio_file')

    duration = float(local_ui_values.get('duration', 5.0))

    if task in ("TA2VA", "IA2VA"):
        if not audio_file:
            raise ValueError(f"Audio file is required for {task} generation.")
        audio_path = save_temp_audio(audio_file)
        local_ui_values['audio_file'] = audio_path
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            duration_seconds = info.duration
        except Exception as e:
            try:
                import mutagen
                audio = mutagen.File(audio_path)
                duration_seconds = audio.info.length
            except Exception as e2:
                print(f"Warning probing audio metadata: {e2}, falling back to duration parameter.")
                duration_seconds = duration

        video_length = int(round(duration_seconds * fps)) + 1
        local_ui_values['video_length'] = video_length
        local_ui_values['audio_duration'] = duration_seconds
    else:
        video_length = int(round(duration * fps)) + 1
        local_ui_values['video_length'] = video_length

    if task in ("I2VA", "IA2VA"):
        if not start_img:
            raise ValueError(f"Start image is required for {task} generation.")
        local_ui_values['start_image'] = save_temp_image(start_img)

    if task == "FLF2VA":
        if not start_img:
            raise ValueError("Start image is required for FLF2VA generation.")
        local_ui_values['start_image'] = save_temp_image(start_img)

        if not end_img:
            raise ValueError("End image is required for FLF2VA generation.")
        local_ui_values['end_image'] = save_temp_image(end_img)

        local_ui_values['strength_first'] = float(local_ui_values.get('strength_first', 0.7))
        local_ui_values['strength_last'] = float(local_ui_values.get('strength_last', 0.7))

    # 6. Filename prefix
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}_{task.lower()}"

    # 7. LoRA processing
    local_ui_values['loras'] = process_lora_inputs(ui_values, 'ltx2_5_lora')

    # 8. Assemble workflow
    assembler = WorkflowAssembler(recipe_path, base_path=RECIPES_DIR)
    final_workflow = assembler.assemble(local_ui_values)

    return final_workflow, None
