import gradio as gr
import re
import json
from PIL import Image
from pymediainfo import MediaInfo

UI_INFO = {
    "main_tab": "Tools",
    "sub_tab": "Media Info",
}

def _parse_a1111_parameters(params_text):
    if not params_text: return {}
    
    neg_prompt_keyword = "Negative prompt:"
    parts = re.split(neg_prompt_keyword, params_text, flags=re.IGNORECASE)
    positive_prompt = parts[0].strip()
    
    params_line = ""
    negative_prompt = ""
    if len(parts) > 1:
        remaining_lines = parts[1].strip().split('\n')
        negative_prompt = remaining_lines[0].strip()
        params_line = "\n".join(remaining_lines[1:])
    else:
        prompt_lines = positive_prompt.split('\n')
        if len(prompt_lines) > 1:
            positive_prompt = prompt_lines[0].strip()
            params_line = "\n".join(prompt_lines[1:])

    param_items = [item.strip() for item in params_line.split(',') if item.strip()]
    return {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "parameters": ", ".join(param_items)
    }

def _get_video_metadata_pymediainfo(filepath):
    try:
        media_info = MediaInfo.parse(filepath)
        for track in media_info.tracks:
            if hasattr(track, 'workflow') and track.workflow:
                return track.workflow
            if hasattr(track, 'prompt') and track.prompt:
                return track.prompt
            if hasattr(track, 'comment') and track.comment:
                return track.comment
        return None
    except Exception as e:
        error_message = str(e)
        if "No such file or directory" in error_message and 'mediainfo' in error_message:
             return "Error: 'mediainfo' not found. Please ensure mediainfo is installed and in your system's PATH environment variable."
        return f"Error extracting video metadata: {e}"


def get_info_from_media(media_type, image_input, video_input):
    a1111_group_update = gr.update(visible=False)
    comfy_group_update = gr.update(visible=False)
    pos_prompt, neg_prompt, gen_params, workflow, raw_params = "", "", "", "", ""
    
    media_file = image_input if media_type == "Image" else video_input

    if not media_file:
        pos_prompt = f"Please upload a {media_type.lower()} file."
        a1111_group_update = gr.update(visible=True)
    else:
        metadata_str = None
        if media_type == "Image":
            info_dict = image_input.info or {}
            metadata_str = info_dict.get("workflow") or info_dict.get("prompt") or info_dict.get("parameters")
        elif media_type == "Video":
            metadata_str = _get_video_metadata_pymediainfo(video_input)

        if not metadata_str:
            pos_prompt = f"No metadata found in the {media_type.lower()} file."
            a1111_group_update = gr.update(visible=True)
        elif metadata_str.startswith("Error:"):
             pos_prompt = metadata_str
             a1111_group_update = gr.update(visible=True)
        else:
            raw_params = metadata_str
            is_a1111 = "Negative prompt:" in metadata_str and "Steps:" in metadata_str
            is_comfy_json = metadata_str.strip().startswith("{") and metadata_str.strip().endswith("}")

            if is_a1111:
                parsed = _parse_a1111_parameters(metadata_str)
                pos_prompt, neg_prompt, gen_params = parsed.get('positive_prompt', ''), parsed.get('negative_prompt', ''), parsed.get('parameters', '')
                a1111_group_update = gr.update(visible=True)
            elif is_comfy_json:
                pos_prompt = "ComfyUI workflow embedded. Please check the Workflow/Prompt box below."
                try:
                    parsed_json = json.loads(metadata_str)
                    workflow = json.dumps(parsed_json, indent=2)
                except (json.JSONDecodeError, TypeError):
                    workflow = metadata_str
                comfy_group_update = gr.update(visible=True)
            else:
                pos_prompt = "Found metadata in an unknown format, displayed below:"
                neg_prompt = metadata_str
                a1111_group_update = gr.update(visible=True)

    return (
        a1111_group_update, comfy_group_update, 
        pos_prompt, neg_prompt, gen_params, 
        workflow, raw_params
    )

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Media Info Extractor")
        gr.Markdown("💡 **Tip:** Upload an image or video. For video metadata parsing, `mediainfo` must be installed on your system.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_type'] = gr.Radio(["Image", "Video"], label="Media Type", value="Image")
                components['input_image'] = gr.Image(type="pil", label="Upload Image", visible=True, height=380)
                components['input_video'] = gr.Video(label="Upload Video", visible=False, height=380)

            with gr.Column(scale=1):
                with gr.Group(visible=True) as a1111_group:
                    components['a1111_outputs_group'] = a1111_group
                    components['info_positive_prompt_output'] = gr.Textbox(label="Prompt", lines=5, interactive=False)
                    components['info_negative_prompt_output'] = gr.Textbox(label="Negative Prompt", lines=3, interactive=False)
                    components['info_generation_params_output'] = gr.Textbox(label="Parameters", lines=3, interactive=False)
                
                with gr.Group(visible=False) as comfy_group:
                    components['comfyui_outputs_group'] = comfy_group
                    components['info_workflow_output'] = gr.Textbox(label="Workflow / Prompt (JSON)", lines=13, interactive=False)
        
        with gr.Row():
            components['copy_info_button'] = gr.Button("📋 Copy Raw Info", variant="primary")
            
    components['raw_params_output'] = gr.Textbox(visible=False)
    return components

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    input_type = components['input_type']
    input_image = components['input_image']
    input_video = components['input_video']
    copy_info_button = components['copy_info_button']
    raw_params_output = components['raw_params_output']
    
    outputs_for_get_info = [
        components['a1111_outputs_group'],
        components['comfyui_outputs_group'],
        components['info_positive_prompt_output'],
        components['info_negative_prompt_output'],
        components['info_generation_params_output'],
        components['info_workflow_output'],
        raw_params_output
    ]

    def update_input_visibility(choice):
        is_image = choice == "Image"
        return gr.update(visible=is_image), gr.update(visible=not is_image)

    input_type.change(
        fn=update_input_visibility,
        inputs=[input_type],
        outputs=[input_image, input_video],
        show_api=False
    )

    def on_upload(media_type, image_file, video_file):
        return get_info_from_media(media_type, image_file, video_file)

    input_image.upload(fn=on_upload, inputs=[input_type, input_image, input_video], outputs=outputs_for_get_info, show_api=False)
    input_video.upload(fn=on_upload, inputs=[input_type, input_image, input_video], outputs=outputs_for_get_info, show_api=False)
    
    def show_copy_notification(text_to_copy: str):
        if text_to_copy and text_to_copy.strip() and not text_to_copy.startswith("Error:"):
            gr.Info("Copied to clipboard!")
        else:
            gr.Warning("No metadata found in the file to copy.")
        return text_to_copy

    copy_js_function = """
    (text) => {
      if (text && text.trim()) {
        const isError = text.startsWith("Error:");
        if (!isError) {
          navigator.clipboard.writeText(text);
        }
      }
      return text;
    }
    """

    copy_info_button.click(
        fn=show_copy_notification,
        inputs=[raw_params_output],
        outputs=[raw_params_output],
        js=copy_js_function,
        show_progress=False,
        show_api=False
    )