import gradio as gr
import os
import traceback
import time

from .newbie_logic import process_inputs
from core.comfy_api import run_workflow_and_get_output

UI_INFO = {
    "main_tab": "Tools",
    "sub_tab": "NewBie XML Generator",
    "run_button_text": "📄 Generate XML"
}

def create_character_ui(components, char_num):
    """Creates UI components for a single character, including an enable checkbox."""
    prefix = f"char{char_num}_"
    with gr.Tab(f"Character {char_num}"):
        with gr.Column():
            components[f'{prefix}enable'] = gr.Checkbox(label="Enable Character", value=(char_num == 1))
            
            components[f'{prefix}name'] = gr.Textbox(label="Name", placeholder="e.g., Hatsune Miku")
            components[f'{prefix}gender'] = gr.Textbox(label="Gender", placeholder="e.g., 1girl", value="1girl")
            components[f'{prefix}appearance'] = gr.Textbox(label="Appearance", placeholder="e.g., aqua hair, twintails, blue eyes", lines=2)
            components[f'{prefix}clothing'] = gr.Textbox(label="Clothing", placeholder="e.g., school uniform, black thighhighs", lines=2)
            components[f'{prefix}expression'] = gr.Textbox(label="Expression", placeholder="e.g., smiling, open mouth")
            components[f'{prefix}action'] = gr.Textbox(label="Action", placeholder="e.g., holding leek, singing")
            components[f'{prefix}interaction'] = gr.Textbox(label="Interaction with others", placeholder="e.g., holding hands with character 2")
            components[f'{prefix}position'] = gr.Textbox(label="Position", placeholder="e.g., in center of image, upper body")

def create_ui():
    """Creates the Gradio UI interface."""
    components = {}
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## NewBie XML Prompt Generator")
            gr.Markdown("💡 **Tip:** Fill in the following fields to generate an XML-formatted prompt for AI art. Use the 'Enable' checkbox under each character tab to decide which characters to include. Requires the `NewBie` series of custom nodes.")

            with gr.Accordion("Global Settings", open=True):
                components['caption'] = gr.Textbox(label="Caption", placeholder="A brief description of the overall scene.")
                components['style_prefix'] = gr.Textbox(label="Style Prefix", placeholder="e.g., masterpiece, best quality")
            
            with gr.Accordion("Characters (up to 4)", open=False):
                with gr.Tabs():
                    for i in range(1, 5):
                        create_character_ui(components, i)

            with gr.Accordion("General Tags", open=False):
                components['count'] = gr.Textbox(label="Count", placeholder="e.g., 1girl, solo", value="1girl, solo")
                components['body_type'] = gr.Textbox(label="Body Type", placeholder="e.g., loli, mature")
                components['artists'] = gr.Textbox(label="Artists", placeholder="e.g., art by Ilya Kuvshinov")
                components['style'] = gr.Textbox(label="Style", placeholder="e.g., anime style", value="anime style")
                components['background'] = gr.Textbox(label="Background", placeholder="e.g., simple background")
                components['environment'] = gr.Textbox(label="Environment", placeholder="e.g., outdoors, night")
                components['perspective'] = gr.Textbox(label="Perspective", placeholder="e.g., from above, cowboy shot")
                components['atmosphere'] = gr.Textbox(label="Atmosphere", placeholder="e.g., emotional, dramatic")
                components['lighting'] = gr.Textbox(label="Lighting", placeholder="e.g., cinematic lighting, rim lighting")
                components['quality'] = gr.Textbox(label="Quality", placeholder="e.g., high quality illustration", value="high quality illustration, clean lineart, no logo, no watermark")
                components['objects'] = gr.Textbox(label="Objects", placeholder="e.g., sword, book")
                components['extra_tags'] = gr.Textbox(label="Extra Tags", placeholder="Any other tags")

        with gr.Column(scale=1):
            components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
            components['xml_output'] = gr.Textbox(
                label="Generated XML", lines=40, interactive=False, show_copy_button=True
            )

    return components

def get_main_output_components(components: dict):
    """Returns the list of main output components."""
    return [components['xml_output'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def run_generation(ui_values):
    """The main function to run the generation task."""
    final_text_content = "Processing..."
    expected_text_file_path = None
    try:
        yield ("Status: Preparing...", "Processing...")
        
        workflow, extra_data = process_inputs(ui_values)
        expected_text_file_path = extra_data.get("expected_text_file_path")
        workflow_package = (workflow, extra_data)
        
        for status, _ in run_workflow_and_get_output(workflow_package):
            yield (status, "Processing...")
        
        yield ("Status: Reading generated XML...", "Processing...")
        
        text_file_found = False
        for _ in range(10): 
            if expected_text_file_path and os.path.exists(expected_text_file_path):
                text_file_found = True
                break
            time.sleep(0.5)

        if text_file_found:
            with open(expected_text_file_path, 'r', encoding='utf-8') as file:
                final_text_content = file.read()
        else:
            final_text_content = "Error: Could not find the generated XML file after processing."

    except Exception as e:
        traceback.print_exc()
        final_text_content = f"An error occurred: {e}"
        yield (f"Error: {e}", final_text_content)
        return
    finally:
        if expected_text_file_path and os.path.exists(expected_text_file_path):
            try:
                os.remove(expected_text_file_path)
                print(f"Cleaned up temporary XML file: {expected_text_file_path}")
            except Exception as e:
                print(f"Error cleaning up temporary XML file: {e}")

    yield ("Status: Loaded successfully!", final_text_content)