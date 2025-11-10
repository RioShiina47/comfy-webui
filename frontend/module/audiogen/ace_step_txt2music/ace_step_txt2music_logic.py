import random
import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "ace_step_txt2music_recipe.yaml"

def process_inputs(params, seed_override=None):
    local_params = params.copy()
    
    if seed_override is not None:
        local_params['seed'] = seed_override
    else:
        seed = int(local_params.get('seed', -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        local_params['seed'] = seed
    
    local_params['filename_prefix'] = f"audio/{get_filename_prefix()}"

    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_params)
    
    return workflow, None