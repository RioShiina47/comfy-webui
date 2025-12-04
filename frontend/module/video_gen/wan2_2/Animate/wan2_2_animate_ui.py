import gradio as gr
import traceback

from .wan2_2_animate_logic import process_inputs, ASPECT_RATIO_PRESETS, PREVIEW_LENGTH
from core.utils import create_batched_run_generation

UI_INFO = {
    "main_tab": "VideoGen",
    "sub_tab": "Animate",
    "run_button_text": "🚀 Animate Video"
}

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

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files[-1] if files else None)
)