import gradio as gr
from .wan2_1_wanmove_logic import process_inputs as process_inputs_logic
from core.utils import create_simple_run_generation, create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "main_tab": "VideoGen",
    "sub_tab": "wan2.1 wanmove",
    "run_button_text": "🌀 Generate"
}

MAX_TRACK_SEGMENTS = 5

RESOLUTION_PRESETS = {
    "720p": {
        "16:9 (Landscape)": (1280, 720),
        "9:16 (Portrait)": (720, 1280),
        "1:1 (Square)": (960, 960),
        "4:3 (Classic TV)": (1088, 816),
        "3:4 (Classic Portrait)": (816, 1088),
        "3:2 (Photography)": (1152, 768),
        "2:3 (Photography Portrait)": (768, 1152),
    },
    "480p": {
        "16:9 (Landscape)": (848, 480),
        "9:16 (Portrait)": (480, 848),
        "1:1 (Square)": (640, 640),
        "4:3 (Classic TV)": (640, 480),
        "3:4 (Classic Portrait)": (480, 640),
        "3:2 (Photography)": (720, 480),
        "2:3 (Photography Portrait)": (480, 720),
    }
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## WanMove Track Visualizer & Generator")
        gr.Markdown("💡 **Tip:** Use 'Preview Tracks' to visualize motion paths. Switch to 'Generate Video' to create the final output. You can use <a href='https://comfyui-wiki.github.io/wan_move_track_visualizer/' target='_blank'>this editor</a> to generate tracks visually.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image", height=294)

            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=4, placeholder="(Optional) Describe the scene.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4)

        with gr.Row():
            with gr.Column(scale=1):
                components['mode'] = gr.Radio(choices=["Preview Tracks", "Generate Video"], value="Preview Tracks", label="Mode")
                with gr.Row():
                    components['resolution'] = gr.Radio(
                        label="Resolution",
                        choices=["480p", "720p"],
                        value="720p",
                        interactive=True
                    )
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(RESOLUTION_PRESETS["720p"].keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                components['num_frames'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=81, step=1, value=81)
                
                with gr.Group(visible=False) as gen_video_only_settings:
                    with gr.Row():
                        components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                        components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)
                components['gen_video_only_settings'] = gen_video_only_settings

            with gr.Column(scale=1):
                components['output_video'] = gr.Video(label="Result", show_label=False, interactive=False, height=381)

        with gr.Accordion("Tracks Settings", open=False):
            components['track_segment_count'] = gr.State(1)
            segment_groups = []
            
            components['all_start_xs'] = []
            components['all_start_ys'] = []
            components['all_end_xs'] = []
            components['all_end_ys'] = []
            components['all_beziers'] = []
            components['all_mid_xs'] = []
            components['all_mid_ys'] = []
            components['all_interpolations'] = []
            components['all_num_tracks'] = []
            components['all_track_spreads'] = []

            for i in range(MAX_TRACK_SEGMENTS):
                with gr.Accordion(f"Track {i+1}", open=(i==0), visible=(i == 0)) as segment_group:
                    with gr.Row():
                        with gr.Column():
                            start_x = gr.Slider(label="Start X", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                            end_x = gr.Slider(label="End X", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                        with gr.Column():
                            start_y = gr.Slider(label="Start Y", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                            end_y = gr.Slider(label="End Y", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                    
                    with gr.Row(visible=True) as bezier_controls_row:
                        with gr.Column():
                            mid_x = gr.Slider(label="Mid X (Bezier)", minimum=0.0, maximum=1.0, step=0.01, value=0.46)
                        with gr.Column():
                            mid_y = gr.Slider(label="Mid Y (Bezier)", minimum=0.0, maximum=1.0, step=0.01, value=0.44)

                    with gr.Row():
                        bezier = gr.Checkbox(label="Use Bezier Curve", value=True)
                        interpolation = gr.Dropdown(label="Interpolation", choices=["linear", "ease_in", "ease_out", "ease_in_out", "constant"], value="linear")
                    
                    with gr.Row():
                        num_tracks = gr.Slider(label="Number of Tracks", minimum=1, maximum=20, step=1, value=5)
                        track_spread = gr.Slider(label="Track Spread", minimum=0.0, maximum=0.1, step=0.001, value=0.007)

                    bezier.change(fn=lambda x: gr.update(visible=x), inputs=[bezier], outputs=[bezier_controls_row], show_api=False)

                    components['all_start_xs'].append(start_x)
                    components['all_start_ys'].append(start_y)
                    components['all_end_xs'].append(end_x)
                    components['all_end_ys'].append(end_y)
                    components['all_beziers'].append(bezier)
                    components['all_mid_xs'].append(mid_x)
                    components['all_mid_ys'].append(mid_y)
                    components['all_interpolations'].append(interpolation)
                    components['all_num_tracks'].append(num_tracks)
                    components['all_track_spreads'].append(track_spread)
                
                segment_groups.append(segment_group)
            
            components['segment_groups'] = segment_groups
            with gr.Row():
                components['add_segment_button'] = gr.Button("✚ Add Track Segment")
                components['delete_segment_button'] = gr.Button("➖ Delete Track Segment", visible=False)

        with gr.Accordion("Visualization Settings (Preview Only)", open=False) as viz_accordion:
            with gr.Row():
                components['line_resolution'] = gr.Slider(label="Line Resolution", minimum=1, maximum=100, step=1, value=24)
                components['circle_size'] = gr.Slider(label="Circle Size", minimum=1, maximum=50, step=1, value=22)
            with gr.Row():
                components['opacity'] = gr.Slider(label="Opacity", minimum=0.0, maximum=1.0, step=0.05, value=0.9)
                components['line_width'] = gr.Slider(label="Line Width", minimum=1, maximum=20, step=1, value=8)
        components['viz_accordion'] = viz_accordion
        
        with gr.Group(visible=False) as lora_group:
            create_lora_ui(components, "wan2_1_wanmove_lora", accordion_label="LoRA Settings")
        components['lora_group'] = lora_group

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "wan2_1_wanmove_lora")
    
    def add_segment(count):
        count += 1
        return (count, gr.update(visible=count < MAX_TRACK_SEGMENTS), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(MAX_TRACK_SEGMENTS))

    def delete_segment(count):
        count -= 1
        return (count, gr.update(visible=True), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(MAX_TRACK_SEGMENTS))

    add_btn = components['add_segment_button']
    del_btn = components['delete_segment_button']
    count_state = components['track_segment_count']
    groups = components['segment_groups']
    
    add_outputs = [count_state, add_btn, del_btn] + groups
    del_outputs = [count_state, add_btn, del_btn] + groups

    add_btn.click(fn=add_segment, inputs=[count_state], outputs=add_outputs, show_api=False)
    del_btn.click(fn=delete_segment, inputs=[count_state], outputs=del_outputs, show_api=False)

    def update_mode_visibility(mode):
        is_preview = (mode == "Preview Tracks")
        return {
            components['viz_accordion']: gr.update(visible=is_preview),
            components['gen_video_only_settings']: gr.update(visible=not is_preview),
            components['lora_group']: gr.update(visible=not is_preview)
        }
    
    components['mode'].change(
        fn=update_mode_visibility,
        inputs=[components['mode']],
        outputs=[components['viz_accordion'], components['gen_video_only_settings'], components['lora_group']],
        show_api=False
    )
    
    def update_aspect_ratio_choices(resolution):
        return gr.update(choices=list(RESOLUTION_PRESETS[resolution].keys()))

    components['resolution'].change(
        fn=update_aspect_ratio_choices,
        inputs=[components['resolution']],
        outputs=[components['aspect_ratio']],
        show_api=False
    )

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    resolution = local_ui_values.get('resolution', '720p')
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Landscape)") 
    width, height = RESOLUTION_PRESETS[resolution][selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    return process_inputs_logic(local_ui_values, seed_override)

def run_generation(ui_values):
    mode = ui_values.get('mode', 'Preview Tracks')
    if mode == 'Generate Video':
        return create_batched_run_generation(
            process_inputs,
            lambda status, files: (status, files[-1] if files else None)
        )(ui_values)
    else:
        return create_simple_run_generation(
            process_inputs,
            lambda status, files: (status, files[-1] if files else None)
        )(ui_values)