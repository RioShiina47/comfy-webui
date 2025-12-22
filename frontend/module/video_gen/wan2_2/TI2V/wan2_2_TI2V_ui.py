import gradio as gr
import traceback
from .wan2_2_TI2V_logic import process_inputs, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "wan2_2_TI2V_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "TI2V",
    "run_button_text": "🎬 Generate TI2V Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.2 TI2V (5B)")
        gr.Markdown("💡 **Tip:** This is the non-lightning version with more controls for generation.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt for video generation...")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)
                
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=121, step=1, value=121)
                
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
        
        create_lora_ui(components, "ti2v_lora", "LoRA Settings")

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "ti2v_lora")

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files[-1] if files else None)
)