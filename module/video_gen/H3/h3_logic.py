import os
import math
from core.workflow_assembler import WorkflowAssembler
from core.utils import get_filename_prefix, handle_seed, save_temp_image, save_temp_video, save_temp_audio
from core.input_processors import process_lora_inputs

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

RESOLUTION_PRESETS = {
    "1080p": {
        "16:9 (Landscape)": (1920, 1088),
        "9:16 (Portrait)": (1088, 1920),
        "1:1 (Square)": (1440, 1440),
        "4:3 (Classic TV)": (1664, 1248),
        "3:4 (Classic Portrait)": (1248, 1664),
        "3:2 (Landscape)": (1760, 1184),
        "2:3 (Portrait)": (1184, 1760),
    },
    "768p": {
        "16:9 (Landscape)": (1344, 768),
        "9:16 (Portrait)": (768, 1344),
        "1:1 (Square)": (1024, 1024),
        "4:3 (Classic TV)": (1152, 864),
        "3:4 (Classic Portrait)": (864, 1152),
        "3:2 (Landscape)": (1248, 832),
        "2:3 (Portrait)": (832, 1248),
    },
    "544p": {
        "16:9 (Landscape)": (960, 544),
        "9:16 (Portrait)": (544, 960),
        "1:1 (Square)": (736, 736),
        "4:3 (Classic TV)": (832, 640),
        "3:4 (Classic Portrait)": (640, 832),
        "3:2 (Landscape)": (864, 576),
        "2:3 (Portrait)": (576, 864),
    }
}

ASPECT_RATIO_PRESETS = RESOLUTION_PRESETS["768p"]

def calculate_h3_frame_length(duration_seconds: float) -> int:
    """
    Converts duration (seconds) at 24fps to a valid frame length
    snapped up to the model's 17-frame-per-block (17k+5) grid.
    Grid sequence: 5, 22, 39, 56, 73, 90, 107, 124, 141...
    """
    raw_frames = int(round(duration_seconds * 24))
    if raw_frames <= 5:
        return 5
    return 5 + 17 * math.ceil((raw_frames - 5) / 17)

def process_h3_guide_inputs(ui_values: dict, prefix: str = "") -> list:
    """
    Parses and prepares MiniMax H3 keyframe guide inputs.
    """
    active_guides = []
    key = lambda name: f"{prefix}_{name}" if prefix else name
    images = ui_values.get(key('h3_guide_images'), [])
    videos = ui_values.get(key('h3_guide_videos'), [])
    audios = ui_values.get(key('h3_guide_audios'), [])
    times = ui_values.get(key('h3_guide_times'), [])
    frames = ui_values.get(key('h3_guide_frames'), [])

    if images or videos or audios:
        max_len = max(len(images), len(videos), len(audios), len(times), len(frames))
        for i in range(max_len):
            img = images[i] if i < len(images) else None
            vid = videos[i] if i < len(videos) else None
            aud = audios[i] if i < len(audios) else None
            t_val = times[i] if i < len(times) else None
            f_val = frames[i] if i < len(frames) else None

            saved_img = save_temp_image(img) if img is not None else None
            saved_vid = save_temp_video(vid) if vid else None
            saved_aud = save_temp_audio(aud) if aud else None

            if not (saved_img or saved_vid or saved_aud):
                continue

            if f_val is not None and str(f_val).strip() != "":
                try:
                    frame_idx = int(round(float(f_val)))
                except (ValueError, TypeError):
                    frame_idx = 0
            elif t_val is not None and str(t_val).strip() != "":
                try:
                    frame_idx = int(round(float(t_val) * 24))
                except (ValueError, TypeError):
                    frame_idx = 0
            else:
                frame_idx = 0

            guide_dict = {"frame_idx": max(0, frame_idx)}
            if saved_img:
                guide_dict["image"] = saved_img
            if saved_vid:
                guide_dict["video"] = saved_vid
            if saved_aud:
                guide_dict["audio"] = saved_aud
            active_guides.append(guide_dict)

    raw_guides = ui_values.get('h3_guides', [])
    if raw_guides and not active_guides and isinstance(raw_guides, list):
        for item in raw_guides:
            if isinstance(item, dict):
                img = item.get('image')
                vid = item.get('video')
                aud = item.get('audio')

                saved_img = save_temp_image(img) if img is not None else None
                saved_vid = save_temp_video(vid) if vid else None
                saved_aud = save_temp_audio(aud) if aud else None

                if not (saved_img or saved_vid or saved_aud):
                    continue

                if item.get('frame_idx') is not None:
                    try:
                        frame_idx = int(round(float(item['frame_idx'])))
                    except (ValueError, TypeError):
                        frame_idx = 0
                elif item.get('time_seconds') is not None or item.get('time') is not None:
                    try:
                        raw_t = item.get('time_seconds') if item.get('time_seconds') is not None else item.get('time')
                        frame_idx = int(round(float(raw_t) * 24))
                    except (ValueError, TypeError):
                        frame_idx = 0
                else:
                    frame_idx = 0

                guide_dict = {"frame_idx": max(0, frame_idx)}
                if saved_img:
                    guide_dict["image"] = saved_img
                if saved_vid:
                    guide_dict["video"] = saved_vid
                if saved_aud:
                    guide_dict["audio"] = saved_aud
                active_guides.append(guide_dict)

    return active_guides

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    task = local_ui_values.get('task', 'FL2VA')

    width = int(local_ui_values.get('width') or 0)
    height = int(local_ui_values.get('height') or 0)
    
    if width <= 0 or height <= 0:
        resolution = local_ui_values.get('resolution', '768p')
        selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Landscape)")
        width, height = RESOLUTION_PRESETS.get(resolution, {}).get(selected_ratio, (1344, 768))
        
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    local_ui_values['steps'] = int(local_ui_values.get('steps') or 20)

    duration = float(local_ui_values.get('duration', 3.0))
    local_ui_values['length'] = calculate_h3_frame_length(duration)
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)
    
    filename_prefix = get_filename_prefix()
    local_ui_values['filename_prefix'] = f"video/{filename_prefix}"

    # 公共 LoRA 与 Keyframe Guide 处理
    local_ui_values['loras'] = process_lora_inputs(ui_values, 'h3_lora')
    local_ui_values['h3_guides'] = process_h3_guide_inputs(ui_values, 'h3')

    # 根据 Task 路由分发
    if task == "REF2VA":
        recipe_path = os.path.join(MODULE_DIR, "workflow_recipes", "h3_ref2va_recipe.yaml")
        local_ui_values['first_frame_loader_class'] = None
        local_ui_values['first_frame_scale_class'] = None
        local_ui_values['last_frame_loader_class'] = None
        local_ui_values['last_frame_scale_class'] = None

        ref_images_input = local_ui_values.get('ref_images', [])
        saved_ref_images = []
        if isinstance(ref_images_input, list):
            for img in ref_images_input:
                if img is not None:
                    saved_ref_images.append(save_temp_image(img))
        local_ui_values['ref_images'] = saved_ref_images

        ref_videos_input = local_ui_values.get('ref_videos', [])
        saved_ref_videos = []
        if isinstance(ref_videos_input, list):
            for video_path in ref_videos_input:
                if video_path:
                    saved_vid = save_temp_video(video_path)
                    if saved_vid:
                        saved_ref_videos.append(saved_vid)
        local_ui_values['ref_videos'] = saved_ref_videos

        ref_audios_input = local_ui_values.get('ref_audios', [])
        saved_ref_audios = []
        if isinstance(ref_audios_input, list):
            for audio_path in ref_audios_input:
                if audio_path:
                    saved_aud = save_temp_audio(audio_path)
                    if saved_aud:
                        saved_ref_audios.append(saved_aud)
        local_ui_values['ref_audios'] = saved_ref_audios
    else:
        recipe_path = os.path.join(MODULE_DIR, "workflow_recipes", "h3_fl2va_recipe.yaml")

        # T2VA 不读取首尾帧；I2VA 只读取首帧；FLF2VA 读取首尾帧
        allow_first_frame = task in ("I2VA", "FLF2VA", "FL2VA")
        allow_last_frame = task in ("FLF2VA", "FL2VA")

        first_frame_img = local_ui_values.get('first_frame') if allow_first_frame else None
        if first_frame_img is not None:
            local_ui_values['first_frame_loader_class'] = "LoadImage"
            local_ui_values['first_frame_scale_class'] = "ImageScale"
            local_ui_values['first_frame_image'] = save_temp_image(first_frame_img)
        else:
            local_ui_values['first_frame_loader_class'] = None
            local_ui_values['first_frame_scale_class'] = None

        last_frame_img = local_ui_values.get('last_frame') if allow_last_frame else None
        if last_frame_img is not None:
            local_ui_values['last_frame_loader_class'] = "LoadImage"
            local_ui_values['last_frame_scale_class'] = "ImageScale"
            local_ui_values['last_frame_image'] = save_temp_image(last_frame_img)
        else:
            local_ui_values['last_frame_loader_class'] = None
            local_ui_values['last_frame_scale_class'] = None

    assembler = WorkflowAssembler(recipe_path, base_path=MODULE_DIR)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None
