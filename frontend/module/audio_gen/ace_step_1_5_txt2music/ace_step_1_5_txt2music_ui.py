import gradio as gr
from core.utils import create_batched_run_generation
from .ace_step_1_5_txt2music_logic import process_inputs
from core import node_info_manager

UI_INFO = {
    "main_tab": "AudioGen",
    "sub_tab": "txt2music (v1.5)",
    "run_button_text": "🎵 Generate Music"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## ACE-Step 1.5 Text-to-Music")
        gr.Markdown("💡 **Tip:** This model is specialized for generating high-quality pop music with lyrics. Use tags like `[Verse]`, `[Chorus]` in lyrics to control structure.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['tags'] = gr.Textbox(
                    label="Tags / Description", 
                    lines=4, 
                    value="",
                    placeholder="Example: Female Vocals, Pop, Dreamy, Synthesizer, 120 BPM.\nDescribe the style, genre, mood, instruments, and vocal characteristics."
                )
                components['lyrics'] = gr.Textbox(
                    label="Lyrics", 
                    lines=10, 
                    value="",
                    placeholder="Example Structure:\n\n[Intro]\n(Instrumental)\n\n[Verse 1]\nWalking down the street\nRhythm in my feet"
                )

            with gr.Column(scale=1):
                keyscale_choices = node_info_manager.get_node_input_options("TextEncodeAceStepAudio1.5", "keyscale")
                language_choices = node_info_manager.get_node_input_options("TextEncodeAceStepAudio1.5", "language")

                with gr.Row():
                    components['clips'] = gr.Dropdown(
                        label="CLIPs", 
                        choices=["0.6b+1.7b", "0.6b+4b"], 
                        value="0.6b+1.7b", 
                        interactive=True
                    )
                    components['duration'] = gr.Slider(label="Duration (seconds)", minimum=5, maximum=300, step=1, value=120)

                with gr.Row():
                    components['language'] = gr.Dropdown(label="Language", choices=language_choices, value="en", interactive=True)
                    components['bpm'] = gr.Number(label="BPM", value=190)

                with gr.Row():
                    components['keyscale'] = gr.Dropdown(label="Key & Scale", choices=keyscale_choices, value="E minor", interactive=True)
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

                with gr.Row():
                    components['batch_size'] = gr.Number(label="Batch Size", value=1, step=1, minimum=1, interactive=True)
                    components['batch_count'] = gr.Number(label="Batch Count", value=1, step=1, minimum=1, interactive=True)

        with gr.Row():
            components['output_audio'] = gr.Audio(label="Result", show_label=True, interactive=False, show_download_button=True)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

        components['steps'] = gr.State(8)
        components['timesignature'] = gr.State("4")
                
    return components

def get_main_output_components(components: dict):
    return [components['output_audio'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)