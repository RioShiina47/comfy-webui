import gradio as gr
from .hunyuanvideo_1_5_img2video_logic import process_inputs, RESOLUTION_PRESETS
from core.utils import create_batched_run_generation

UI_INFO = {
    "main_tab": "VideoGen",
    "sub_tab": "HunyuanVideo-1.5 I2V",
    "run_button_text": "🎬 Generate Video from Image"
}

RESOLUTION_CHOICES = list(RESOLUTION_PRESETS.keys())
ASPECT_RATIO_CHOICES = list(RESOLUTION_PRESETS["480p"].keys())

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## HunyuanVideo-1.5 Image-to-Video")
        gr.Markdown("💡 **Tip:** Upload an image and describe the desired motion. Higher resolutions build upon the 480p result and will take longer.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Start Image", height=294)
            
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Prompt (Motion Description)", lines=4, placeholder="Describe the motion or change you want to see...")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4, value="")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['resolution'] = gr.Dropdown(
                        label="Resolution",
                        choices=RESOLUTION_CHOICES,
                        value=RESOLUTION_CHOICES[0],
                        interactive=True
                    )
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=ASPECT_RATIO_CHOICES,
                        value=ASPECT_RATIO_CHOICES[0],
                        interactive=True
                    )
                with gr.Row():
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=121, step=1, value=121)
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                with gr.Row():
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)
                    components['use_easy_cache'] = gr.Checkbox(label="Use EasyCache", value=True)

            with gr.Column(scale=1):
                components['output_video'] = gr.Gallery(
                    label="Result", 
                    show_label=False,
                    interactive=False,
                    height=395,
                    object_fit="contain",
                    columns=2,
                    preview=True
                )
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)