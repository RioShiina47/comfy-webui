import gradio as gr
import random
import os
import yaml
import shutil
from PIL import Image
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe_char_replacement": "wan2_2_animate_char_replacement_recipe.yaml",
    "workflow_recipe_pose_transfer": "wan2_2_animate_pose_transfer_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "Animate",
    "run_button_text": "🚀 Animate Video"
}

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

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.2 Animate")
        gr.Markdown("💡 **Tip:** Upload a reference image for the character/style and a video for the motion. Use 'Preview' for a quick test.")
        
        with gr.Row():
            components['mode'] = gr.Radio(
                choices=["Character Replacement", "Pose Transfer"],
                value="Character Replacement",
                label="Animate Mode",
                info="Choose 'Character Replacement' to replace the person in the video, or 'Pose Transfer' to only transfer the pose."
            )
            components['gen_mode'] = gr.Radio(
                choices=[f"Preview ({PREVIEW_LENGTH} frames)", "Generate Full Video"],
                value=f"Preview ({PREVIEW_LENGTH} frames)",
                label="Generation Mode"
            )

        with gr.Row():
            components['ref_image'] = gr.Image(type="pil", label="Reference Image (Character/Style)", scale=1, height=380)
            components['motion_video'] = gr.Video(label="Motion Video", scale=1, height=380)

        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the character and scene.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3, value="")
                
                with gr.Group(visible=True) as sam_group:
                    components['sam_prompt'] = gr.Textbox(label="SAM Segmentation Prompt", placeholder="e.g., human, person, man, woman. Leave empty to default to 'human'.", info="Describe the object to be replaced in the video.")
                components['sam_group'] = sam_group

                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="Auto (from video)",
                        interactive=True
                    )
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)
            
            with gr.Column(scale=1):
                components['output_video'] = gr.Video(label="Result", show_label=False, interactive=False, height=472)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    def update_mode_visibility(mode):
        is_char_replacement = (mode == "Character Replacement")
        return gr.update(visible=is_char_replacement)
    
    components['mode'].change(
        fn=update_mode_visibility,
        inputs=[components['mode']],
        outputs=[components['sam_group']],
        show_api=False
    )

def save_temp_file(file_obj, name_prefix: str, is_video=False) -> str:
    if file_obj is None: return None
    comfyui_input_dir = COMFYUI_INPUT_PATH
    
    if is_video:
        ext = os.path.splitext(file_obj)[1] if os.path.splitext(file_obj)[1] else ".mp4"
        temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}{ext}"
        save_path = os.path.join(comfyui_input_dir, temp_filename)
        shutil.copy(file_obj, save_path)
    else:
        temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}.png"
        save_path = os.path.join(comfyui_input_dir, temp_filename)
        file_obj.save(save_path, "PNG")
        
    print(f"Saved temporary input file to: {save_path}")
    return temp_filename

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    ref_image_pil = local_ui_values.get('ref_image')
    motion_video_path = local_ui_values.get('motion_video')

    if ref_image_pil is None: raise gr.Error("Reference image is required.")
    if motion_video_path is None: raise gr.Error("Motion video is required.")

    metadata = get_media_metadata(motion_video_path, is_video=True)
    duration = metadata.get('duration', 0)
    fps = metadata.get('fps', 24)
    total_video_frames = int(duration * fps)

    if not total_video_frames or total_video_frames <= 0:
        raise gr.Error("Could not determine video length from duration and FPS.")
    
    selected_ratio_key = local_ui_values.get('aspect_ratio')
    if selected_ratio_key == "Auto (from video)":
        width, height = metadata['width'], metadata['height']
        if width == 0 or height == 0:
            raise gr.Error("Could not auto-detect video dimensions. Please select a specific aspect ratio.")
    else:
        width, height = ASPECT_RATIO_PRESETS[selected_ratio_key]
    
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    local_ui_values['ref_image'] = save_temp_file(ref_image_pil, "animate_ref", is_video=False)
    local_ui_values['motion_video'] = save_temp_file(motion_video_path, "animate_motion", is_video=True)
    
    current_seed = 0
    if seed_override is not None:
        current_seed = seed_override
    else:
        seed = int(local_ui_values.get('seed', -1))
        if seed == -1: seed = random.randint(0, 2**32 - 1)
        current_seed = seed
    local_ui_values['seed'] = current_seed

    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    animate_mode = local_ui_values.get('mode')
    gen_mode = local_ui_values.get('gen_mode', f"Preview ({PREVIEW_LENGTH} frames)")

    if animate_mode == "Character Replacement":
        recipe_path = UI_INFO["workflow_recipe_char_replacement"]
        sam_prompt = local_ui_values.get('sam_prompt', '').strip()
        local_ui_values['sam_prompt'] = sam_prompt if sam_prompt else "human"
    else:
        recipe_path = UI_INFO["workflow_recipe_pose_transfer"]
    
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

def run_generation(ui_values):
    all_output_files = []
    
    try:
        batch_count = int(ui_values.get('batch_count', 1))
        original_seed = int(ui_values.get('seed', -1))

        for i in range(batch_count):
            current_seed = original_seed + i if original_seed != -1 else None
            batch_msg = f" (Batch {i + 1}/{batch_count})" if batch_count > 1 else ""
            
            yield (f"Status: Preparing{batch_msg}...", None)
            
            workflow, extra_data = process_inputs(ui_values, seed_override=current_seed)
            workflow_package = (workflow, extra_data)
            
            for status, output_path in run_workflow_and_get_output(workflow_package):
                status_msg = f"Status: {status.replace('Status: ', '')}{batch_msg}"
                
                final_video = None
                if output_path and isinstance(output_path, list):
                    all_output_files.extend(p for p in output_path if p not in all_output_files)
                    final_video = all_output_files[-1]

                yield (status_msg, final_video)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return
        
    yield ("Status: Loaded successfully!", all_output_files[-1] if all_output_files else None)