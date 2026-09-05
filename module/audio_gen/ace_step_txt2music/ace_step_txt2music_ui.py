import gradio as gr
import random
import os
import traceback
from .ace_step_txt2music_logic import process_inputs
from core.utils import create_simple_run_generation

UI_INFO = {
    "workflow_recipe": "ace_step_txt2music_recipe.yaml",
    "main_tab": "AudioGen",
    "sub_tab": "txt2music",
    "run_button_text": "🎵 Generate Music"
}

DEFAULT_LYRICS = """[instrumental]
[break down]
[drum fill]
[chopped samples]
"""

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## ACE-Step Text-to-Music")
        gr.Markdown("💡 **Tip:** To generate instrumental music, keep the default lyrics. To generate a song, enter your own lyrics in English.")
        
        with gr.Row():
            with gr.Column(scale=2):
                components['tags'] = gr.Textbox(
                    label="Tags", 
                    lines=3, 
                    placeholder="Describe the music style, genre, instruments, mood, etc. (e.g., epic, cinematic, powerful, orchestral, dramatic, intense)"
                )
                components['lyrics'] = gr.Textbox(
                    label="Lyrics", 
                    lines=5, 
                    value=DEFAULT_LYRICS
                )
                
                with gr.Row():
                    components['seconds'] = gr.Slider(label="Duration (seconds)", minimum=5, maximum=300, step=1, value=30)
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                
                components['steps'] = gr.State(50)
                components['cfg'] = gr.State(5.0)
                components['sampler_name'] = gr.State("euler")
                components['scheduler'] = gr.State("simple")

            with gr.Column(scale=1):
                components['output_audio'] = gr.Audio(label="Result", show_label=False, interactive=False)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_audio'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_simple_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)