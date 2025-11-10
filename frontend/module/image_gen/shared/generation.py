import traceback
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix

def create_run_generation_logic(process_inputs_func, ui_info, prefix):
    def run_generation(ui_values):
        all_files = []
        
        try:
            batch_count = int(ui_values.get(f'{prefix}_batch_count', 1))
            seed = int(ui_values.get(f'{prefix}_seed', -1))
            
            for i in range(batch_count):
                current_seed = seed + i if seed != -1 else None
                
                ui_values_with_prefix = ui_values.copy()
                ui_values_with_prefix[f'{prefix}_filename_prefix'] = get_filename_prefix()

                yield (f"Status: Preparing batch {i + 1}/{batch_count}...", all_files)
                
                workflow, extra_data = process_inputs_func(ui_values_with_prefix, seed_override=current_seed)
                workflow_package = (workflow, extra_data)

                for status, output_path in run_workflow_and_get_output(workflow_package):
                    if output_path and isinstance(output_path, list):
                        new_files = [f for f in output_path if f not in all_files]
                        if new_files:
                            all_files.extend(new_files)
                    
                    batch_status = f"Status: [Batch {i+1}/{batch_count}] {status.replace('Status: ', '')}"
                    yield (batch_status, all_files)

        except Exception as e:
            traceback.print_exc()
            yield (f"Error: {e}", all_files)
            return

        yield ("Status: Loaded successfully!", all_files)
            
    return run_generation