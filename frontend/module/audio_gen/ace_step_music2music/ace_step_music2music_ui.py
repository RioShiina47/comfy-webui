import gradio as gr
import random
import os
import shutil
from .ace_step_music2music_logic import process_inputs
from core.utils import create_simple_run_generation

UI_INFO = {
    "workflow_recipe": "ace_step_music2music_recipe.yaml",
    "main_tab": "AudioGen",
    "sub_tab": "music2music",
    "run_button_text": "🎼 Re-compose Music"
}

DEFAULT_LYRICS = """[instrumental]
[break down]
[drum fill]
[chopped samples]
"""

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## ACE-Step Music-to-Music")
        gr.Markdown("💡 **Tip:** Keep the default lyrics to generate instrumental music; enter your own lyrics to generate a song.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_audio'] = gr.Audio(type="filepath", label="Input Audio")
                
                components['similarity'] = gr.Slider(
                    label="Similarity", 
                    minimum=0.0, 
                    maximum=1.0, 
                    step=0.05, 
                    value=0.7,
                    info="Higher value = more similar to original audio"
                )
                
            with gr.Column(scale=2):
                components['tags'] = gr.Textbox(
                    label="Tags", 
                    lines=3, 
                    placeholder="Describe the desired changes in style, genre, instruments, etc."
                )
                components['lyrics'] = gr.Textbox(
                    label="Lyrics", 
                    lines=5, 
                    value=DEFAULT_LYRICS
                )
                components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

            with gr.Column(scale=1):
                components['output_audio'] = gr.Audio(label="Result", show_label=False, interactive=False, show_download_button=True)

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