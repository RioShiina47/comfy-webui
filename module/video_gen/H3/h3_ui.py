import gradio as gr
from .h3_logic import process_inputs, RESOLUTION_PRESETS
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "workflow_recipes/h3_fl2va_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "H3",
    "run_button_text": "🎬 Generate H3 Video"
}

MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_H3_GUIDES = 5

TASK_CHOICES = ["T2VA", "I2VA", "FLF2VA", "REF2VA"]

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## MiniMax H3 Video Generation")
        
        components['task'] = gr.Radio(
            choices=TASK_CHOICES,
            value="T2VA",
            label="Task",
            interactive=True
        )

        with gr.Row(visible=False) as frame_row:
            components['first_frame'] = gr.Image(type="pil", label="First Frame", height=220, visible=False)
            components['last_frame'] = gr.Image(type="pil", label="Last Frame", height=220, visible=False)
        components['frame_row'] = frame_row

        components['prompt'] = gr.Textbox(label="Prompt", lines=5)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['resolution'] = gr.Radio(
                        label="Resolution",
                        choices=["544p", "768p", "1080p"],
                        value="768p",
                        interactive=True
                    )

                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(RESOLUTION_PRESETS["768p"].keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )

                with gr.Row():
                    components['width'] = gr.Number(label="Width", value=1344, precision=0)
                    components['height'] = gr.Number(label="Height", value=768, precision=0)
                
                with gr.Row():
                    components['steps'] = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=20,
                        interactive=True
                    )
                    components['duration'] = gr.Slider(
                        label="Duration (s)",
                        minimum=0.2,
                        maximum=15.0,
                        step=0.1,
                        value=5.0,
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

        create_lora_ui(components, "h3_lora", accordion_label="LoRA Settings")

        # Multi-modal Reference Panels (REF2VA)
        with gr.Accordion("Reference Image Settings", open=False, visible=False) as ref_img_accordion:
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
        components['ref_img_accordion'] = ref_img_accordion

        with gr.Accordion("Reference Video Settings", open=False, visible=False) as ref_vid_accordion:
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
        components['ref_vid_accordion'] = ref_vid_accordion

        with gr.Accordion("Reference Audio Settings", open=False, visible=False) as ref_aud_accordion:
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
        components['ref_aud_accordion'] = ref_aud_accordion

        # Keyframe Guide Panel (Shared)
        with gr.Accordion("Keyframe Guide Settings", open=False):
            gr.Markdown(
                "💡 **Tip:** You can anchor keyframe guides (image, video clip, or audio) "
                "at arbitrary positions along the continuous timeline (not limited to first/last frames). "
                "Specify the position via **Time (seconds)**."
            )
            h3_guide_rows = []
            h3_guide_images = []
            h3_guide_videos = []
            h3_guide_audios = []
            h3_guide_times = []

            for i in range(MAX_H3_GUIDES):
                with gr.Row(visible=(i < 1)) as row:
                    with gr.Column(scale=1):
                        h3_guide_images.append(gr.Image(label=f"Guide Image {i+1}", type="pil", sources=["upload"], height=200))
                    with gr.Column(scale=1):
                        h3_guide_videos.append(gr.Video(label=f"Guide Video (Opt) {i+1}", sources=["upload"], height=200))
                    with gr.Column(scale=1):
                        h3_guide_audios.append(gr.Audio(label=f"Guide Audio (Opt) {i+1}", sources=["upload"], type="filepath"))
                        h3_guide_times.append(gr.Slider(label=f"Time (s) {i+1}", minimum=0.0, maximum=15.0, step=0.5, value=float(i * 1.5), interactive=True))
                    h3_guide_rows.append(row)

            components['h3_guide_rows'] = h3_guide_rows
            components['h3_guide_images'] = h3_guide_images
            components['h3_guide_videos'] = h3_guide_videos
            components['h3_guide_audios'] = h3_guide_audios
            components['h3_guide_times'] = h3_guide_times

            with gr.Row():
                components['add_h3_guide_button'] = gr.Button("✚ Add Guide")
                components['delete_h3_guide_button'] = gr.Button("➖ Delete Guide", visible=True)
            components['h3_guide_count_state'] = gr.State(1)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "h3_lora")

    def on_task_change(task_val):
        is_i2va = (task_val == "I2VA")
        is_flf2va = (task_val in ("FLF2VA", "FL2VA"))
        is_ref2va = (task_val == "REF2VA")
        show_frame_row = is_i2va or is_flf2va

        return (
            gr.update(visible=show_frame_row),           # frame_row
            gr.update(visible=show_frame_row),           # first_frame
            gr.update(visible=is_flf2va),                # last_frame
            gr.update(visible=is_ref2va),                # ref_img_accordion
            gr.update(visible=is_ref2va),                # ref_vid_accordion
            gr.update(visible=is_ref2va)                 # ref_aud_accordion
        )

    components['task'].change(
        fn=on_task_change,
        inputs=[components['task']],
        outputs=[
            components['frame_row'],
            components['first_frame'],
            components['last_frame'],
            components['ref_img_accordion'],
            components['ref_vid_accordion'],
            components['ref_aud_accordion']
        ],
        api_name=False
    )

    def update_dimensions(resolution, aspect_ratio):
        w, h = RESOLUTION_PRESETS.get(resolution, {}).get(aspect_ratio, (1344, 768))
        return w, h

    components['resolution'].change(
        fn=update_dimensions,
        inputs=[components['resolution'], components['aspect_ratio']],
        outputs=[components['width'], components['height']],
        api_name=False
    )

    components['aspect_ratio'].change(
        fn=update_dimensions,
        inputs=[components['resolution'], components['aspect_ratio']],
        outputs=[components['width'], components['height']],
        api_name=False
    )

    # Reference Image events
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
        api_name=False
    )

    del_ref_btn.click(
        fn=delete_ref_row,
        inputs=[ref_count_state],
        outputs=del_ref_outputs,
        show_progress=False,
        api_name=False
    )

    # Reference Video events
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
        api_name=False
    )

    del_ref_vid_btn.click(
        fn=delete_ref_video_row,
        inputs=[ref_video_count_state],
        outputs=del_ref_vid_outputs,
        show_progress=False,
        api_name=False
    )

    # Reference Audio events
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
        api_name=False
    )

    del_ref_aud_btn.click(
        fn=delete_ref_audio_row,
        inputs=[ref_audio_count_state],
        outputs=del_ref_aud_outputs,
        show_progress=False,
        api_name=False
    )

    # Keyframe Guide events
    guide_count_state = components['h3_guide_count_state']
    add_guide_btn = components['add_h3_guide_button']
    del_guide_btn = components['delete_h3_guide_button']
    guide_rows = components['h3_guide_rows']
    guide_images = components['h3_guide_images']
    guide_videos = components['h3_guide_videos']
    guide_audios = components['h3_guide_audios']

    def add_guide_row(count):
        count += 1
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_H3_GUIDES))
        return (count, gr.update(visible=count < MAX_H3_GUIDES), gr.update(visible=count > 0)) + visibility_updates

    def delete_guide_row(count):
        count -= 1
        img_clears = [gr.update()] * MAX_H3_GUIDES
        vid_clears = [gr.update()] * MAX_H3_GUIDES
        aud_clears = [gr.update()] * MAX_H3_GUIDES
        if count >= 0:
            img_clears[count] = None
            vid_clears[count] = None
            aud_clears[count] = None
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_H3_GUIDES))
        return (count, gr.update(visible=count < MAX_H3_GUIDES), gr.update(visible=count > 0)) + visibility_updates + tuple(img_clears) + tuple(vid_clears) + tuple(aud_clears)

    add_guide_outputs = [guide_count_state, add_guide_btn, del_guide_btn] + guide_rows
    del_guide_outputs = [guide_count_state, add_guide_btn, del_guide_btn] + guide_rows + guide_images + guide_videos + guide_audios

    add_guide_btn.click(
        fn=add_guide_row,
        inputs=[guide_count_state],
        outputs=add_guide_outputs,
        show_progress=False,
        api_name=False
    )

    del_guide_btn.click(
        fn=delete_guide_row,
        inputs=[guide_count_state],
        outputs=del_guide_outputs,
        show_progress=False,
        api_name=False
    )

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)
