import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image, save_temp_video

WORKFLOW_RECIPE_CHAR_REPLACEMENT = "wan2_2_animate_char_replacement_recipe.yaml"
WORKFLOW_RECIPE_POSE_TRANSFER = "wan2_2_animate_pose_transfer_recipe.yaml"

ASPECT_RATIO_PRESETS = {
    "Auto (from video)": None,
    "16:9 (Landscape)": (1280, 720),
    "9:16 (Portrait)": (720, 1280),
    "1:1 (Square)": (960, 960),
    "4:3 (Classic TV)": (1088, 816),
    "3:4 (Classic Portrait)": (816, 1088),
    "3:2 (Photography)": (1152, 768),
    "2:3 (Photography Portrait)": (768, 1152),
}
PREVIEW_LENGTH = 77

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    ref_image_pil = local_ui_values.get('ref_image')
    motion_video_path = local_ui_values.get('motion_video')

    if ref_image_pil is None: raise ValueError("Reference image is required.")
    if motion_video_path is None: raise ValueError("Motion video is required.")

    metadata = get_media_metadata(motion_video_path, is_video=True)
    duration = metadata.get('duration', 0)
    fps = metadata.get('fps', 24)
    total_video_frames = int(duration * fps)

    if not total_video_frames or total_video_frames <= 0:
        raise ValueError("Could not determine video length from duration and FPS.")
    
    selected_ratio_key = local_ui_values.get('aspect_ratio')
    if selected_ratio_key == "Auto (from video)":
        width, height = metadata['width'], metadata['height']
        if width == 0 or height == 0:
            raise ValueError("Could not auto-detect video dimensions. Please select a specific aspect ratio.")
    else:
        width, height = ASPECT_RATIO_PRESETS[selected_ratio_key]
    
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    local_ui_values['ref_image'] = save_temp_image(ref_image_pil)
    local_ui_values['motion_video'] = save_temp_video(motion_video_path)
    
    current_seed = handle_seed(seed_override if seed_override is not None else local_ui_values.get('seed', -1))
    local_ui_values['seed'] = current_seed
    
    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    animate_mode = local_ui_values.get('mode')
    gen_mode = local_ui_values.get('gen_mode', f"Preview ({PREVIEW_LENGTH} frames)")

    if animate_mode == "Character Replacement":
        recipe_path = WORKFLOW_RECIPE_CHAR_REPLACEMENT
        sam_prompt = local_ui_values.get('sam_prompt', '').strip()
        local_ui_values['sam_prompt'] = sam_prompt if sam_prompt else "human"
    else:
        recipe_path = WORKFLOW_RECIPE_POSE_TRANSFER
    
    if gen_mode == 'Generate Full Video':
        local_ui_values['wan_animate_chain'] = {
            "video_length": total_video_frames,
            "seed": current_seed,
            "mode": animate_mode
        }
    else:
        local_ui_values['video_length'] = min(total_video_frames, PREVIEW_LENGTH)

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)

    if gen_mode != 'Generate Full Video':
        template_node_ids_to_remove = {
            v for k, v in assembler.node_map.items() if k.startswith('template_')
        }
        
        for node_id in template_node_ids_to_remove:
            if node_id in final_workflow:
                del final_workflow[node_id]
        print(f"Preview mode: Cleaned up {len(template_node_ids_to_remove)} template nodes.")

    return final_workflow, None