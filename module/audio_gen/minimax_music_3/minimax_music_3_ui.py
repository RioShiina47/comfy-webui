import gradio as gr
from core.utils import create_batched_run_generation
from .minimax_music_3_logic import process_inputs

UI_INFO = {
    "workflow_recipe": "minimax_music_3_recipe.yaml",
    "main_tab": "AudioGen",
    "sub_tab": "txt2music(music3)",
    "run_button_text": "🎵 Generate Music"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## MiniMax Music 3")
        gr.Markdown("💡 **Tip:** Enter a caption describing the music style, mood, and genre. Provide lyrics with section tags (e.g., `[Intro]`, `[Verse]`, `[Chorus]`) or leave empty for instrumental.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['caption'] = gr.Textbox(
                    label="Caption / Style Description", 
                    lines=4, 
                    value="",
                    placeholder="Example: Female Vocals, Pop ballad, Emotional, Piano, Strings, 120 BPM, high quality studio recording."
                )
                components['lyrics'] = gr.Textbox(
                    label="Lyrics", 
                    lines=10, 
                    value="",
                    placeholder="Example Structure:\n\n[Intro]\n(Instrumental)\n\n[Verse 1]\nStars begin to shine tonight\nGuiding through the darkest night\n\n[Chorus]\nFly away beyond the sky\nFeel the dream and never die"
                )

            with gr.Column(scale=1):
                with gr.Row():
                    components['max_duration'] = gr.Slider(label="Duration (seconds)", minimum=5, maximum=300, step=1, value=60)
                    components['cfg_scale'] = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.7)

                with gr.Row():
                    components['top_k'] = gr.Slider(label="Top-K", minimum=1, maximum=100, step=1, value=50)
                    components['steps'] = gr.Slider(label="Sampling Steps", minimum=10, maximum=100, step=1, value=30)

                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

                with gr.Row():
                    components['batch_size'] = gr.Number(label="Batch Size", value=1, step=1, minimum=1, interactive=True)
                    components['batch_count'] = gr.Number(label="Batch Count", value=1, step=1, minimum=1, interactive=True)

        with gr.Row():
            components['output_audio'] = gr.Audio(label="Result", show_label=True, interactive=False)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

        components['sampler_name'] = gr.State("euler")
        components['scheduler'] = gr.State("simple")
        components['tile_size'] = gr.State(1536)
        components['overlap'] = gr.State(64)
                
    return components

def get_main_output_components(components: dict):
    return [components['output_audio'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)
