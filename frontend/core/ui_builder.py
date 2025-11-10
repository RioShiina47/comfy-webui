import gradio as gr
import os
import time
from core import job_manager

from core.backend_manager import backend_manager

def build_gradio_ui(demo: gr.Blocks, ui_tree: dict, ui_modules: dict, layout_config: dict, share_mode: bool):
    all_components = {}
    modules_with_handlers = []
    module_component_map = {}

    ordered_main_tabs = layout_config.get("main_tabs_order", [])
    ordered_sub_tabs_map = layout_config.get("sub_tabs_order", {})

    if share_mode:
        print("Share mode is enabled. Disabling the History tab for privacy.")
        if "History" in ui_tree:
            del ui_tree["History"]
        ordered_main_tabs = [tab for tab in ordered_main_tabs if (isinstance(tab, str) and tab != "History") or (isinstance(tab, dict) and next(iter(tab)) != "History")]

    discovered_tabs = list(ui_tree.keys())
    final_tab_order = []
    
    for tab in ordered_main_tabs:
        if tab in discovered_tabs:
            final_tab_order.append(tab)
            discovered_tabs.remove(tab)
    final_tab_order.extend(sorted(discovered_tabs))

    with gr.Tabs():
        for main_tab_name in final_tab_order:
            sub_tab_infos = ui_tree.get(main_tab_name, [])
            
            ordered_sub_tabs_structure = ordered_sub_tabs_map.get(main_tab_name, [])
            discovered_sub_tabs_map = {info['sub_tab']: info for info in sub_tab_infos}

            def build_final_structure(layout_structure, discovered_map):
                final_list = []
                for item in layout_structure:
                    if isinstance(item, dict):
                        group_name = next(iter(item))
                        nested_list = build_final_structure(item[group_name], discovered_map)
                        if nested_list:
                            final_list.append({group_name: nested_list})
                    else:
                        sub_tab_name = item
                        if sub_tab_name in discovered_map:
                            final_list.append(discovered_map.pop(sub_tab_name))
                return final_list

            final_nested_infos = build_final_structure(ordered_sub_tabs_structure, discovered_sub_tabs_map)
            
            for sub_tab_name in sorted(discovered_sub_tabs_map.keys()):
                final_nested_infos.append(discovered_sub_tabs_map[sub_tab_name])
            
            if not final_nested_infos:
                continue

            with gr.TabItem(main_tab_name):
                build_ui_for_modules(final_nested_infos, ui_modules, all_components, module_component_map, modules_with_handlers)
    
    return all_components, module_component_map, modules_with_handlers


def build_ui_for_modules(nested_infos, ui_modules, all_components, module_component_map, modules_with_handlers):
    is_simple_module = (len(nested_infos) == 1 and 
                        isinstance(nested_infos[0], dict) and 
                        "sub_tab" in nested_infos[0] and 
                        nested_infos[0]['sub_tab'] == nested_infos[0]['main_tab'])

    if is_simple_module:
        info = nested_infos[0]
        module = ui_modules[info["sub_tab"]]
        _create_and_bind_module_ui(module, all_components, module_component_map, modules_with_handlers)
    else:
        with gr.Tabs():
            for item in nested_infos:
                if "sub_tab" in item:
                    with gr.TabItem(item["sub_tab"]):
                        module = ui_modules[item["sub_tab"]]
                        _create_and_bind_module_ui(module, all_components, module_component_map, modules_with_handlers)
                else:
                    group_name = next(iter(item))
                    with gr.TabItem(group_name):
                        build_ui_for_modules(item[group_name], ui_modules, all_components, module_component_map, modules_with_handlers)


def _create_and_bind_module_ui(module, all_components, module_component_map, modules_with_handlers):
    recovered_job = job_manager.get_latest_running_job_for_module(module.__name__)
    initial_job_id = None
    initial_polling_val = None
    initial_status_msg = "Status: Ready"

    if recovered_job:
        print(f"[UI Builder] Found recovered job {recovered_job['id']} for module {module.__name__}")
        initial_job_id = recovered_job['id']
        initial_polling_val = str(time.time())
        initial_status_msg = recovered_job.get("progress_message", "Status: Recovering...")
    
    job_id_state = gr.State(value=initial_job_id)
    polling_trigger = gr.Textbox(value=initial_polling_val, visible=False, label="Polling Trigger")
    status_bar = gr.Textbox(value=initial_status_msg, label="Status", interactive=False, show_label=False, container=False)
    last_status_message_state = gr.State(initial_status_msg)

    components = module.create_ui()
    
    components.update({
        'job_id_state': job_id_state,
        'polling_trigger': polling_trigger,
        'status_bar': status_bar,
        'last_status_message_state': last_status_message_state
    })
    
    all_components.update(components)
    module_component_map[module.__name__] = components

    if hasattr(module, "create_event_handlers"):
        modules_with_handlers.append(module)
    
    if hasattr(module, "run_generation") and hasattr(module, "get_main_output_components"):
        run_button = components.get('run_button')
        if not run_button:
            print(f"Warning: Module {module.__name__} has run_generation but no 'run_button' in components.")
            return

        flat_inputs, input_keys = _collect_module_inputs(components)
        main_outputs = module.get_main_output_components(components)

        submit_job, check_job_status = _define_job_functions(components, input_keys, main_outputs, module)

        buttons_to_bind = [run_button] if not isinstance(run_button, list) else run_button
        for btn in buttons_to_bind:
            btn.click(
                fn=submit_job,
                inputs=flat_inputs, 
                outputs=[job_id_state, polling_trigger, last_status_message_state, status_bar],
                show_api=False
            )

        polling_trigger.change(
            fn=check_job_status,
            inputs=[job_id_state, polling_trigger, last_status_message_state],
            outputs=[status_bar] + main_outputs + [polling_trigger, last_status_message_state],
            show_progress="hidden",
            show_api=False
        )

def _collect_module_inputs(components):
    flat_inputs = []
    input_keys = []
    for key, comp in components.items():
        is_input_type = isinstance(comp, (gr.State, gr.Textbox, gr.Slider, gr.Dropdown, gr.Number, gr.Checkbox, gr.Radio, gr.Image, gr.Video, gr.Audio, gr.UploadButton, gr.ImageEditor))
        is_input_list = isinstance(comp, list) and all(isinstance(c, (gr.State, gr.Textbox, gr.Slider, gr.Dropdown, gr.Number, gr.Checkbox, gr.UploadButton, gr.Image)) for c in comp)
        
        if (is_input_type or is_input_list) and 'output_' not in key and not key.startswith('info_') and key not in ['run_button', 'job_id_state', 'polling_trigger', 'status_bar', 'last_status_message_state']:
            input_keys.append(key)
            if is_input_list:
                flat_inputs.extend(comp)
            else:
                flat_inputs.append(comp)
    return flat_inputs, input_keys

def _define_job_functions(components, input_keys, main_outputs, module):
    def submit_job(*args):
        target_backend = module.UI_INFO.get("target_backend", "default")
        
        yield (gr.update(), gr.update(), gr.update(), f"Status: Switching to '{target_backend}' backend...")

        backend_manager.switch_backend(target_backend)
        
        ui_values = {}
        arg_index = 0
        for key in input_keys:
            component = components[key]
            if isinstance(component, list):
                num_items = len(component)
                ui_values[key] = list(args[arg_index : arg_index + num_items])
                arg_index += num_items
            else:
                ui_values[key] = args[arg_index]
                arg_index += 1
        
        job_id = job_manager.create_job(ui_values, module)
        job_manager.run_job_in_background(job_id)
        
        yield job_id, str(time.time()), "Status: Ready", "Status: Task queued..."

    def check_job_status(job_id, polling_val, last_status_message):
        if not job_id:
            return (gr.update(),) * (1 + len(main_outputs)) + (gr.update(every=None), gr.update())

        job = job_manager.get_job(job_id)
        status_message = job.get("progress_message") or job.get("error_message", "Status: Unknown")
        result_files = job.get("result_files") or []
        
        status_update = gr.update()
        if status_message != last_status_message:
            status_update = status_message
        
        IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
        VIDEO_EXTS = {'.mp4', '.webm', '.mkv', '.mov'}
        AUDIO_EXTS = {'.mp3', '.wav', '.flac'}
        MODEL3D_EXTS = {'.glb', '.obj'}

        outputs = [[] for _ in range(5)]
        ext_map = {0: IMAGE_EXTS, 1: VIDEO_EXTS, 2: AUDIO_EXTS, 3: MODEL3D_EXTS}
        for f in result_files:
            if not f: continue
            ext = os.path.splitext(f)[1].lower()
            found = False
            for i, exts in ext_map.items():
                if ext in exts:
                    outputs[i].append(f)
                    found = True
                    break
            if not found:
                outputs[4].append(f)
        
        output_updates = []
        image_files = sorted(outputs[0])
        video_files = sorted(outputs[1])
        audio_files = sorted(outputs[2])
        model3d_files = sorted(outputs[3])
        other_files = sorted(outputs[4])
        
        for comp in main_outputs:
            if isinstance(comp, gr.Button): 
                output_updates.append(gr.update())
            elif isinstance(comp, gr.Gallery): 
                gallery_files = image_files + video_files
                output_updates.append(gallery_files if gallery_files else gr.update())
            elif isinstance(comp, gr.Video):
                target_file = None
                if hasattr(comp, 'label') and comp.label:
                    if 'mask' in comp.label.lower() and any('mask' in f for f in video_files):
                        target_file = next((f for f in video_files if 'mask' in f), None)
                    elif 'result' in comp.label.lower() and any('result' in f for f in video_files):
                         target_file = next((f for f in video_files if 'result' in f), None)
                
                if target_file:
                    video_files.remove(target_file)
                    output_updates.append(target_file)
                elif video_files:
                    output_updates.append(video_files.pop(0))
                else:
                    output_updates.append(gr.update())
            elif isinstance(comp, gr.Audio): 
                output_updates.append(audio_files.pop(0) if audio_files else gr.update())
            elif isinstance(comp, gr.Model3D): 
                output_updates.append(model3d_files.pop(0) if model3d_files else gr.update())
            elif isinstance(comp, gr.Image): 
                output_updates.append(image_files.pop(0) if image_files else gr.update())
            else:
                all_files_remaining = video_files + audio_files + model3d_files + image_files + other_files
                output_updates.append(all_files_remaining.pop(0) if all_files_remaining else gr.update())

        final_updates = [status_update] + output_updates
        
        if job["status"] in [job_manager.STATUS_COMPLETED, job_manager.STATUS_FAILED]:
            button_update = gr.update(value=module.UI_INFO.get("run_button_text", "Generate"), variant="primary")
            polling_update = gr.update(every=None)
        else:
            button_update = gr.update(value="Stop", variant="stop")
            polling_update = gr.update(value=str(time.time()), every=1)

        for i, comp in enumerate(main_outputs):
            if isinstance(comp, gr.Button):
                final_updates[i + 1] = button_update

        final_updates.extend([polling_update, status_message])
        return tuple(final_updates)

    return submit_job, check_job_status