import gradio as gr
from core.utils import create_simple_run_generation
from .ace_step_1_5_music2music_logic import process_inputs
from core import node_info_manager

UI_INFO = {
    "main_tab": "AudioGen",
    "sub_tab": "music2music (v1.5)",
    "run_button_text": "🎼 Re-compose Music"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## ACE-Step 1.5 Music-to-Music")
        gr.Markdown("💡 **Tip:** Vary the output based on an input audio reference. Lower similarity = more creative changes.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_audio'] = gr.Audio(type="filepath", label="Input Audio")
                
                components['similarity'] = gr.Slider(
                    label="Similarity (1 - Denoise)", 
                    minimum=0.0, 
                    maximum=1.0, 
                    step=0.05, 
                    value=0.7,
                    info="Higher value = closer to original audio"
                )

                keyscale_choices = node_info_manager.get_node_input_options("TextEncodeAceStepAudio1.5", "keyscale")
                language_choices = node_info_manager.get_node_input_options("TextEncodeAceStepAudio1.5", "language")

                with gr.Row():
                    components['clips'] = gr.Dropdown(
                        label="CLIPs", 
                        choices=["0.6b+1.7b", "0.6b+4b"], 
                        value="0.6b+1.7b", 
                        interactive=True
                    )

                with gr.Row():
                    components['language'] = gr.Dropdown(label="Language", choices=language_choices, value="en", interactive=True)
                    components['bpm'] = gr.Number(label="BPM", value=190)

                with gr.Row():
                    components['keyscale'] = gr.Dropdown(label="Key & Scale", choices=keyscale_choices, value="E minor", interactive=True)
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

            with gr.Column(scale=1):
                components['tags'] = gr.Textbox(
                    label="Tags / Description", 
                    lines=4, 
                    value="",
                    placeholder="Example: Remix into a lo-fi hip hop beat, relax, chill, study music.\nDescribe the target style, genre, and instruments."
                )
                components['lyrics'] = gr.Textbox(
                    label="Lyrics", 
                    lines=25, 
                    value="",
                    placeholder="Example Structure:\n\n[Intro]\n(Instrumental)\n\n[Verse]\n(Enter lyrics to align with the input audio melody/rhythm)\n\n[Chorus]\n(Sing along)\n"
                )

        with gr.Row():
            components['output_audio'] = gr.Audio(label="Result", show_label=True, interactive=False, show_download_button=True)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
        
        components['timesignature'] = gr.State("4")
                
    return components

def get_main_output_components(components: dict):
    return [components['output_audio'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_simple_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)