import gradio as gr
from .h3_ref2va_logic import process_inputs, RESOLUTION_PRESETS
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "h3_ref2va_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "H3 REF2VA",
    "run_button_text": "🎬 Generate H3 Video"
}

MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## MiniMax H3 Reference Video Generation")
        gr.Markdown("💡 **Tip:** Supports uploading up to 9 reference images, 3 reference videos with audio, and 3 reference audio files.")
        
        components['prompt'] = gr.Textbox(label="Prompt", lines=5)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['resolution'] = gr.Radio(
                        label="Resolution",
                        choices=["480p", "720p", "1080p"],
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

                with gr.Row():
                    components['width'] = gr.Number(label="Width", value=1344, precision=0)
                    components['height'] = gr.Number(label="Height", value=768, precision=0)
                with gr.Row():
                    components['duration'] = gr.Slider(
                        label="Duration (seconds)",
                        minimum=0.2,
                        maximum=15.0,
                        step=0.1,
                        value=3.0,
                        interactive=True
                    )

                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Gallery(
                    label="Result", 
                    show_label=False, 
                    interactive=False, 
                    height=492,
                    object_fit="contain",
                    columns=2,
                    preview=True
                )

        create_lora_ui(components, "h3_ref2va_lora", accordion_label="LoRA Settings")

        with gr.Accordion("Reference Image Settings", open=False):
            ref_image_groups = []
            ref_image_inputs = []
            with gr.Row():
                for i in range(MAX_REF_IMAGES):
                    with gr.Column(visible=(i < 1), min_width=160) as img_col:
                        img_comp = gr.Image(
                            type="pil", 
                            label=f"Ref Image {i+1}", 
                            sources=["upload"], 
                            height=160
                        )
                        ref_image_groups.append(img_col)
                        ref_image_inputs.append(img_comp)
            components['ref_image_groups'] = ref_image_groups
            components['ref_image_inputs'] = ref_image_inputs
            components['ref_images'] = ref_image_inputs
            
            with gr.Row():
                components['add_ref_button'] = gr.Button("✚ Add Reference Image")
                components['delete_ref_button'] = gr.Button("➖ Delete Reference Image", visible=True)
            components['ref_count_state'] = gr.State(1)

        with gr.Accordion("Reference Video Settings", open=False):
            ref_video_groups = []
            ref_video_inputs = []
            with gr.Row():
                for i in range(MAX_REF_VIDEOS):
                    with gr.Column(visible=(i < 1), min_width=200) as vid_col:
                        vid_comp = gr.Video(
                            label=f"Ref Video {i+1}", 
                            sources=["upload"], 
                            height=200
                        )
                        ref_video_groups.append(vid_col)
                        ref_video_inputs.append(vid_comp)
            components['ref_video_groups'] = ref_video_groups
            components['ref_video_inputs'] = ref_video_inputs
            components['ref_videos'] = ref_video_inputs
            
            with gr.Row():
                components['add_ref_video_button'] = gr.Button("✚ Add Reference Video")
                components['delete_ref_video_button'] = gr.Button("➖ Delete Reference Video", visible=True)
            components['ref_video_count_state'] = gr.State(1)

        with gr.Accordion("Reference Audio Settings", open=False):
            ref_audio_groups = []
            ref_audio_inputs = []
            with gr.Row():
                for i in range(MAX_REF_AUDIOS):
                    with gr.Column(visible=(i < 1), min_width=200) as aud_col:
                        aud_comp = gr.Audio(
                            label=f"Ref Audio {i+1}", 
                            sources=["upload"], 
                            type="filepath"
                        )
                        ref_audio_groups.append(aud_col)
                        ref_audio_inputs.append(aud_comp)
            components['ref_audio_groups'] = ref_audio_groups
            components['ref_audio_inputs'] = ref_audio_inputs
            components['ref_audios'] = ref_audio_inputs
            
            with gr.Row():
                components['add_ref_audio_button'] = gr.Button("✚ Add Reference Audio")
                components['delete_ref_audio_button'] = gr.Button("➖ Delete Reference Audio", visible=True)
            components['ref_audio_count_state'] = gr.State(1)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "h3_ref2va_lora")

    def update_dimensions(resolution, aspect_ratio):
        w, h = RESOLUTION_PRESETS.get(resolution, {}).get(aspect_ratio, (1344, 768))
        return w, h

    components['resolution'].change(
        fn=update_dimensions,
        inputs=[components['resolution'], components['aspect_ratio']],
        outputs=[components['width'], components['height']],
        show_api=False
    )

    components['aspect_ratio'].change(
        fn=update_dimensions,
        inputs=[components['resolution'], components['aspect_ratio']],
        outputs=[components['width'], components['height']],
        show_api=False
    )

    ref_count_state = components['ref_count_state']
    add_ref_btn = components['add_ref_button']
    del_ref_btn = components['delete_ref_button']
    ref_image_groups = components['ref_image_groups']
    ref_image_inputs = components['ref_image_inputs']

    def add_ref_row(count):
        count += 1
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_REF_IMAGES))
        return (count, gr.update(visible=count < MAX_REF_IMAGES), gr.update(visible=count > 0)) + visibility_updates

    def delete_ref_row(count):
        count -= 1
        image_clear_updates = [gr.update()] * MAX_REF_IMAGES
        if count >= 0:
            image_clear_updates[count] = None
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_REF_IMAGES))
        return (count, gr.update(visible=count < MAX_REF_IMAGES), gr.update(visible=count > 0)) + visibility_updates + tuple(image_clear_updates)

    add_ref_outputs = [ref_count_state, add_ref_btn, del_ref_btn] + ref_image_groups
    del_ref_outputs = [ref_count_state, add_ref_btn, del_ref_btn] + ref_image_groups + ref_image_inputs

    add_ref_btn.click(
        fn=add_ref_row,
        inputs=[ref_count_state],
        outputs=add_ref_outputs,
        show_progress=False,
        show_api=False
    )

    del_ref_btn.click(
        fn=delete_ref_row,
        inputs=[ref_count_state],
        outputs=del_ref_outputs,
        show_progress=False,
        show_api=False
    )

    ref_video_count_state = components['ref_video_count_state']
    add_ref_vid_btn = components['add_ref_video_button']
    del_ref_vid_btn = components['delete_ref_video_button']
    ref_video_groups = components['ref_video_groups']
    ref_video_inputs = components['ref_video_inputs']

    def add_ref_video_row(count):
        count += 1
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_REF_VIDEOS))
        return (count, gr.update(visible=count < MAX_REF_VIDEOS), gr.update(visible=count > 0)) + visibility_updates

    def delete_ref_video_row(count):
        count -= 1
        video_clear_updates = [gr.update()] * MAX_REF_VIDEOS
        if count >= 0:
            video_clear_updates[count] = None
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_REF_VIDEOS))
        return (count, gr.update(visible=count < MAX_REF_VIDEOS), gr.update(visible=count > 0)) + visibility_updates + tuple(video_clear_updates)

    add_ref_vid_outputs = [ref_video_count_state, add_ref_vid_btn, del_ref_vid_btn] + ref_video_groups
    del_ref_vid_outputs = [ref_video_count_state, add_ref_vid_btn, del_ref_vid_btn] + ref_video_groups + ref_video_inputs

    add_ref_vid_btn.click(
        fn=add_ref_video_row,
        inputs=[ref_video_count_state],
        outputs=add_ref_vid_outputs,
        show_progress=False,
        show_api=False
    )

    del_ref_vid_btn.click(
        fn=delete_ref_video_row,
        inputs=[ref_video_count_state],
        outputs=del_ref_vid_outputs,
        show_progress=False,
        show_api=False
    )

    ref_audio_count_state = components['ref_audio_count_state']
    add_ref_aud_btn = components['add_ref_audio_button']
    del_ref_aud_btn = components['delete_ref_audio_button']
    ref_audio_groups = components['ref_audio_groups']
    ref_audio_inputs = components['ref_audio_inputs']

    def add_ref_audio_row(count):
        count += 1
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_REF_AUDIOS))
        return (count, gr.update(visible=count < MAX_REF_AUDIOS), gr.update(visible=count > 0)) + visibility_updates

    def delete_ref_audio_row(count):
        count -= 1
        audio_clear_updates = [gr.update()] * MAX_REF_AUDIOS
        if count >= 0:
            audio_clear_updates[count] = None
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_REF_AUDIOS))
        return (count, gr.update(visible=count < MAX_REF_AUDIOS), gr.update(visible=count > 0)) + visibility_updates + tuple(audio_clear_updates)

    add_ref_aud_outputs = [ref_audio_count_state, add_ref_aud_btn, del_ref_aud_btn] + ref_audio_groups
    del_ref_aud_outputs = [ref_audio_count_state, add_ref_aud_btn, del_ref_aud_btn] + ref_audio_groups + ref_audio_inputs

    add_ref_aud_btn.click(
        fn=add_ref_audio_row,
        inputs=[ref_audio_count_state],
        outputs=add_ref_aud_outputs,
        show_progress=False,
        show_api=False
    )

    del_ref_aud_btn.click(
        fn=delete_ref_audio_row,
        inputs=[ref_audio_count_state],
        outputs=del_ref_aud_outputs,
        show_progress=False,
        show_api=False
    )

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)
