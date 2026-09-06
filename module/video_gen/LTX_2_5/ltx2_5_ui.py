import gradio as gr
from .ltx2_5_logic import process_inputs, RESOLUTION_PRESETS
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "workflow_recipes/ltx2_5_t2va_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "LTX-2.5",
    "run_button_text": "🎬 Generate LTX-2.5 Video"
}

REQUIRED_LORA_DIRS = ["ltx-2.5", "ltx-2"]
TASK_CHOICES = ["T2VA", "I2VA", "TA2VA", "IA2VA", "FLF2VA"]
DEFAULT_NEG_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"


def create_ui():
    components = {}

    with gr.Column():
        gr.Markdown("## Lightricks LTX-2.5 Video Generation")
        gr.Markdown("💡 **Tip:** Select a generation task below. Supports text, image, audio, and first/last frame transitions.")

        components['task'] = gr.Radio(
            choices=TASK_CHOICES,
            value="T2VA",
            label="Task",
            interactive=True
        )

        # --------------------------------------------------------------------
        # Layout Option A: Side-by-Side (Used for I2VA and TA2VA)
        # Left: Media (Start Image or Audio) | Right: Prompt & Negative Prompt
        # --------------------------------------------------------------------
        with gr.Row(visible=False) as side_container:
            with gr.Column(scale=1):
                components['i2va_start_image'] = gr.Image(type="pil", label="Start Image", height=294, visible=False)
                components['ta2va_audio_file'] = gr.Audio(type="filepath", label="Input Audio", visible=False)
            with gr.Column(scale=2):
                components['side_positive_prompt'] = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the scene, motion, and action...")
                components['side_negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4, value=DEFAULT_NEG_PROMPT)
        components['side_container'] = side_container

        # --------------------------------------------------------------------
        # Layout Option B: Stacked / Two-Row (Used for T2VA, IA2VA, FLF2VA)
        # Row 1: Media (Start Image, End Image, Audio)
        # Row 2: Full-width Prompt & Negative Prompt
        # --------------------------------------------------------------------
        with gr.Column(visible=True) as stacked_container:
            with gr.Row(visible=False) as stacked_media_row:
                components['stacked_start_image'] = gr.Image(type="pil", label="Start Image", scale=1, height=294, visible=False)
                components['stacked_end_image'] = gr.Image(type="pil", label="End Image", scale=1, height=294, visible=False)
                components['stacked_audio_file'] = gr.Audio(type="filepath", label="Input Audio", scale=1, visible=False)
            components['stacked_media_row'] = stacked_media_row

            components['full_positive_prompt'] = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the desired content and motion...")
            components['full_negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4, value=DEFAULT_NEG_PROMPT)
        components['stacked_container'] = stacked_container

        # --------------------------------------------------------------------
        # Common Parameters Grid
        # --------------------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['resolution'] = gr.Radio(
                        label="Resolution",
                        choices=["544p", "768p", "1080p"],
                        value="768p",
                        interactive=True
                    )
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(RESOLUTION_PRESETS['768p'].keys()),
                        value="16:9 (Widescreen)",
                        interactive=True
                    )
                with gr.Row():
                    components['duration'] = gr.Slider(
                        label="Duration (seconds)",
                        minimum=1.0,
                        maximum=20.0,
                        step=1.0,
                        value=5.0,
                        interactive=True
                    )
                    components['fps'] = gr.Dropdown(
                        label="FPS",
                        choices=["24fps", "25fps"],
                        value="24fps",
                        interactive=True
                    )

                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

                with gr.Row():
                    components['use_spatial_upscaler'] = gr.Checkbox(label="Use 2x Spatial Upscaler", value=False, interactive=True)
                    components['use_temporal_upscaler'] = gr.Checkbox(label="Use 2x Temporal Upscaler", value=False, interactive=True)

            with gr.Column(scale=1):
                components['output_video'] = gr.Gallery(
                    label="Result",
                    show_label=False,
                    interactive=False,
                    object_fit="contain",
                    columns=2,
                    preview=True,
                    height=460
                )

        create_lora_ui(components, "ltx2_5_lora", required_lora_dirs=REQUIRED_LORA_DIRS)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components


def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]


def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "ltx2_5_lora")

    def on_task_change(task_val, curr_side_pos, curr_side_neg, curr_full_pos, curr_full_neg, curr_i2va_img, curr_stacked_img, curr_ta2va_aud, curr_stacked_aud):
        is_side = task_val in ("I2VA", "TA2VA")
        is_i2va = (task_val == "I2VA")
        is_ta2va = (task_val == "TA2VA")

        is_stacked = not is_side
        is_ia2va = (task_val == "IA2VA")
        is_flf2va = (task_val == "FLF2VA")

        # Sync prompt texts across layouts seamlessly
        active_pos = curr_side_pos if curr_side_pos else (curr_full_pos or "")
        active_neg = curr_side_neg if curr_side_neg else (curr_full_neg or DEFAULT_NEG_PROMPT)

        # Sync media values across layouts
        active_img = curr_i2va_img if curr_i2va_img is not None else curr_stacked_img
        active_aud = curr_ta2va_aud if curr_ta2va_aud is not None else curr_stacked_aud

        return (
            gr.update(visible=is_side),                                 # side_container
            gr.update(visible=is_i2va, value=active_img),               # i2va_start_image
            gr.update(visible=is_ta2va, value=active_aud),              # ta2va_audio_file
            gr.update(value=active_pos),                                # side_positive_prompt
            gr.update(value=active_neg),                                # side_negative_prompt
            gr.update(visible=is_stacked),                              # stacked_container
            gr.update(visible=is_ia2va or is_flf2va),                   # stacked_media_row
            gr.update(visible=is_ia2va or is_flf2va, value=active_img), # stacked_start_image
            gr.update(visible=is_flf2va),                               # stacked_end_image
            gr.update(visible=is_ia2va, value=active_aud),              # stacked_audio_file
            gr.update(value=active_pos),                                # full_positive_prompt
            gr.update(value=active_neg),                                # full_negative_prompt
        )

    components['task'].change(
        fn=on_task_change,
        inputs=[
            components['task'],
            components['side_positive_prompt'],
            components['side_negative_prompt'],
            components['full_positive_prompt'],
            components['full_negative_prompt'],
            components['i2va_start_image'],
            components['stacked_start_image'],
            components['ta2va_audio_file'],
            components['stacked_audio_file'],
        ],
        outputs=[
            components['side_container'],
            components['i2va_start_image'],
            components['ta2va_audio_file'],
            components['side_positive_prompt'],
            components['side_negative_prompt'],
            components['stacked_container'],
            components['stacked_media_row'],
            components['stacked_start_image'],
            components['stacked_end_image'],
            components['stacked_audio_file'],
            components['full_positive_prompt'],
            components['full_negative_prompt'],
        ],
        api_name=False
    )

    def update_aspect_ratio_choices(resolution):
        resolution_key = str(resolution).lower()
        preset_dict = RESOLUTION_PRESETS.get(resolution_key, RESOLUTION_PRESETS["768p"])
        return gr.update(choices=list(preset_dict.keys()))

    components['resolution'].change(
        fn=update_aspect_ratio_choices,
        inputs=[components['resolution']],
        outputs=[components['aspect_ratio']],
        api_name=False
    )


run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)
