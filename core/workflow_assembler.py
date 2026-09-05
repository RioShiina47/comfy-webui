import yaml
import os
import importlib
import importlib.util
from copy import deepcopy
import re
import sys

from . import node_info_manager
from .yaml_loader import load_and_merge_yaml, load_and_merge_yaml_from_module, ROOT_DIR


class WorkflowAssembler:
    _global_injectors_cache = {}

    def __init__(self, recipe, injector_order: list = None, base_path: str = None, dynamic_values: dict = None):
        self.base_path = base_path
        self.node_counter = 0
        self.workflow = {}
        self.node_map = {}
        self.loaded_local_injectors = {}
        self.injector_order = injector_order or []

        if isinstance(recipe, (str, os.PathLike)):
            recipe_path = str(recipe)
            if not os.path.isabs(recipe_path):
                if not os.path.exists(recipe_path) and base_path:
                    recipe_path = os.path.join(base_path, recipe_path)
            with open(recipe_path, "r", encoding="utf-8") as f:
                content = f.read()
            if dynamic_values:
                for k, v in dynamic_values.items():
                    if v is not None:
                        content = content.replace(f"{{{{ {k} }}}}", str(v))
            self.recipe = yaml.safe_load(content)
        elif isinstance(recipe, dict):
            self.recipe = recipe
        else:
            raise TypeError(f"WorkflowAssembler expects a recipe dict or file path, but got {type(recipe)}.")

    def _get_injector_function(self, chain_type: str):
        if chain_type in self.loaded_local_injectors:
            return self.loaded_local_injectors[chain_type]

        feature_name = chain_type[8:-7] if (chain_type.startswith('dynamic_') and chain_type.endswith('_chains')) else chain_type
        injector_filename = f"{feature_name}_injector.py"

        if self.base_path and os.path.isdir(self.base_path):
            local_file_path = None
            for root, _, files in os.walk(self.base_path):
                if injector_filename in files:
                    local_file_path = os.path.join(root, injector_filename)
                    break

            if local_file_path:
                try:
                    spec = importlib.util.spec_from_file_location(f"{feature_name}_injector", local_file_path)
                    module = importlib.util.module_from_spec(spec)
                    
                    original_sys_path = sys.path[:]
                    local_dir = os.path.dirname(local_file_path)
                    if local_dir not in sys.path:
                        sys.path.insert(0, local_dir)
                    if self.base_path not in sys.path:
                        sys.path.insert(0, self.base_path)
                    
                    spec.loader.exec_module(module)
                    sys.path[:] = original_sys_path

                    if hasattr(module, 'inject') and callable(module.inject):
                        print(f"Dynamically loaded local injector: {local_file_path}")
                        target_chain_type = getattr(module, 'CHAIN_TYPE', chain_type)
                        self.loaded_local_injectors[chain_type] = module.inject
                        if target_chain_type != chain_type:
                            self.loaded_local_injectors[target_chain_type] = module.inject
                        return module.inject
                except Exception as e:
                    print(f"Error loading local injector {local_file_path}: {e}")

        if chain_type in WorkflowAssembler._global_injectors_cache:
            return WorkflowAssembler._global_injectors_cache[chain_type]

        global_injector_dir = os.path.join(ROOT_DIR, "chain_injectors")
        global_file_path = os.path.join(global_injector_dir, injector_filename)

        if os.path.exists(global_file_path):
            mod_name = f"{feature_name}_injector"
            rel_path = f"chain_injectors.{mod_name}"
            try:
                if ROOT_DIR not in sys.path:
                    sys.path.insert(0, ROOT_DIR)
                module = importlib.import_module(rel_path)
                target_chain_type = getattr(module, 'CHAIN_TYPE', chain_type)
                if hasattr(module, 'inject') and callable(module.inject):
                    func = module.inject
                    WorkflowAssembler._global_injectors_cache[target_chain_type] = func
                    print(f"Successfully registered global injector (lazy): {target_chain_type} from {rel_path}")
                    return func
                else:
                    print(f"Warning: Module '{rel_path}' for injector '{chain_type}' does not have an 'inject' function.")
            except Exception as e:
                print(f"Error importing module '{rel_path}' for injector '{chain_type}': {e}")

        return None

    def _get_unique_id(self):
        self.node_counter += 1
        return str(self.node_counter)

    def _get_node_template_from_api(self, class_type):
        node_info = node_info_manager.get_node_info(class_type)
        if not node_info:
            raise ValueError(f"Node with class_type '{class_type}' not found in ComfyUI's /object_info. Is the node installed and named correctly?")

        template = { "inputs": {}, "class_type": class_type, "_meta": { "title": node_info.get("display_name", class_type) } }
        all_inputs = {}
        all_inputs.update(node_info.get("input", {}).get("required", {}))
        all_inputs.update(node_info.get("input", {}).get("optional", {}))
        for name, details in all_inputs.items():
            config = details[1] if len(details) > 1 and isinstance(details[1], dict) else {}
            template["inputs"][name] = config.get("default", None)
        return template

    def assemble(self, ui_values):
        for name, details in self.recipe['nodes'].items():
            if 'class_type' not in details:
                raise KeyError(f"Node '{name}' in recipe is missing the required 'class_type' field.")
            class_type = details['class_type']
            match = re.search(r"\{\{\s*(\w+)\s*\}\}", class_type)
            if match:
                placeholder_key = match.group(1)
                if placeholder_key in ui_values and ui_values[placeholder_key] is not None: 
                    class_type = ui_values[placeholder_key]
                else: 
                    print(f"Warning: Missing or None value for placeholder '{placeholder_key}' in ui_values for class_type '{details['class_type']}'. Skipping node '{name}'.")
                    continue
            template = self._get_node_template_from_api(class_type)
            node_data = deepcopy(template)
            unique_id = self._get_unique_id()
            self.node_map[name] = unique_id
            if 'title' in details: node_data['_meta']['title'] = details['title']
            if 'params' in details:
                for param, value in details['params'].items():
                    if param in node_data['inputs']: node_data['inputs'][param] = value
                    else: print(f"Warning: Param '{param}' in recipe for node '{name}' does not exist in '{class_type}'. Skipping.")
            self.workflow[unique_id] = node_data

        for ui_key, target in self.recipe.get('ui_map', {}).items():
            if ui_key in ui_values and ui_values[ui_key] is not None:
                if ui_key == "vae_loader" and "vae_encode" not in self.node_map:
                    continue
                if isinstance(target, dict) and isinstance(ui_values[ui_key], dict):
                    for sub_key, sub_target in target.items():
                        if sub_key in ui_values[ui_key]:
                            target_name, target_param = sub_target.split(':')
                            if target_name in self.node_map:
                                self.workflow[self.node_map[target_name]]['inputs'][target_param] = ui_values[ui_key][sub_key]
                elif isinstance(target, dict):
                    for sub_key, sub_target in target.items():
                        if sub_key in ui_values:
                            targets = sub_target if isinstance(sub_target, list) else [sub_target]
                            for t in targets:
                                target_name, target_param = t.split(':')
                                if target_name in self.node_map:
                                    self.workflow[self.node_map[target_name]]['inputs'][target_param] = ui_values[sub_key]
                else:
                    target_list = target if isinstance(target, list) else [target]
                    for t in target_list:
                        if isinstance(t, str) and ':' in t:
                            target_name, target_param = t.split(':')
                            if target_name in self.node_map:
                                self.workflow[self.node_map[target_name]]['inputs'][target_param] = ui_values[ui_key]
                        else:
                            print(f"Warning: Skipping invalid target format in ui_map for key '{ui_key}': {t}")
        
        for conn in self.recipe.get('connections', []):
            if not isinstance(conn.get('to'), str) or not isinstance(conn.get('from'), str):
                print(f"Warning: Skipping invalid connection format in recipe: {conn}")
                continue
            from_name, from_output_idx = conn['from'].split(':')
            to_name, to_input_name = conn['to'].split(':')
            from_id, to_id = self.node_map.get(from_name), self.node_map.get(to_name)
            if from_id and to_id: self.workflow[to_id]['inputs'][to_input_name] = [from_id, int(from_output_idx)]
        
        recipe_chain_types = {key for key in self.recipe if key.startswith('dynamic_')}

        processing_order = [key for key in self.injector_order if key in recipe_chain_types]
        
        remaining_chains = sorted(list(recipe_chain_types - set(processing_order)))
        processing_order.extend(remaining_chains)
        if remaining_chains:
            print(f"[WorkflowAssembler] Processing modular injector chains not in global order: {remaining_chains}")

        for chain_type in processing_order:
            injector_func = self._get_injector_function(chain_type)
            if injector_func:
                for chain_key, chain_def in self.recipe.get(chain_type, {}).items():
                    if chain_key in ui_values and ui_values[chain_key]:
                        chain_items = ui_values[chain_key]
                        injector_func(self, chain_def, chain_items)

        return self.workflow