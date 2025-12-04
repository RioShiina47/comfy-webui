import gradio as gr
import traceback
from .wan2_2_sound2video_logic import process_inputs, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation

UI_INFO = {
    "workflow_recipe": "wan2_2_sound2video_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "sound2video",
    "run_button_text": "🔊 Generate from Sound"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.2 S2V (Sound-to-Video) (14B) - Lightning")
        gr.Markdown("💡 **Tip:** Upload a reference image and an audio file. Use 'Preview' for a quick test, or 'Generate Full Video' for the final output.")
        
        components['mode'] = gr.Radio(
            choices=["Preview (77 frames)", "Generate Full Video"],
            value="Preview (77 frames)",
            label="Generation Mode"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                components['ref_image'] = gr.Image(type="pil", label="Reference Image", height=282)
            with gr.Column(scale=1):
                components['audio_file'] = gr.Audio(type="filepath", label="Audio File")

        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the character and scene.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=list(ASPECT_RATIO_PRESETS.keys()),
                    value="16:9 (Landscape)",
                    interactive=True
                )
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Video(
                    label="Result", 
                    show_label=False, 
                    interactive=False, 
                    height=468
                )

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files[-1] if files else None)
)