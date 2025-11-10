import yaml
import os
import importlib
import importlib.util
from copy import deepcopy
import re
import sys

from . import node_info_manager
from .yaml_loader import load_and_merge_yaml

FRONTEND_DIR = os.path.dirname(os.path.dirname(__file__))
BASE_RECIPE_DIR = os.path.join(FRONTEND_DIR, "module", "image_gen", "workflow_recipes")
CUSTOM_RECIPE_DIR = os.path.join(FRONTEND_DIR, "custom", "workflow_recipes")


class WorkflowAssembler:
    def __init__(self, recipe_path, dynamic_values=None, base_path=None):
        self.base_path = base_path
        self.node_counter = 0
        self.workflow = {}
        self.node_map = {}
        self.loaded_local_injectors = {}
        
        self._load_injector_config()

        self.recipe = self._load_and_merge_recipe(recipe_path, dynamic_values or {})
    
    def _load_injector_config(self):
        try:
            injector_config = load_and_merge_yaml("injectors.yaml")
            definitions = injector_config.get("injector_definitions", {})
            self.injector_order = injector_config.get("injector_order", [])
            self.global_injectors = {}

            for chain_type, config in definitions.items():
                module_path = config.get("module")
                if not module_path:
                    print(f"Warning: Injector '{chain_type}' in injectors.yaml is missing 'module' path.")
                    continue
                try:
                    module = importlib.import_module(module_path)
                    if hasattr(module, 'inject'):
                        self.global_injectors[chain_type] = module.inject
                        print(f"Successfully registered global injector: {chain_type} from {module_path}")
                    else:
                        print(f"Warning: Module '{module_path}' for injector '{chain_type}' does not have an 'inject' function.")
                except ImportError as e:
                    print(f"Error importing module '{module_path}' for injector '{chain_type}': {e}")
            
            if not self.injector_order:
                 print("Warning: 'injector_order' is not defined in injectors.yaml. Using definition order.")
                 self.injector_order = list(definitions.keys())

        except Exception as e:
            print(f"FATAL: Could not load or parse injectors.yaml. Dynamic chains will not work. Error: {e}")
            self.injector_order = []
            self.global_injectors = {}


    def _load_and_merge_recipe(self, recipe_filename, dynamic_values, search_context_dir=None):
        normalized_filename = os.path.normpath(recipe_filename)
        
        search_paths = []
        if search_context_dir:
            search_paths.append(os.path.join(search_context_dir, normalized_filename))
        if self.base_path:
            search_paths.append(os.path.join(self.base_path, normalized_filename))
        search_paths.append(os.path.join(CUSTOM_RECIPE_DIR, normalized_filename))
        search_paths.append(os.path.join(BASE_RECIPE_DIR, normalized_filename))
        
        recipe_path_to_use = None
        for path in search_paths:
            if os.path.exists(path):
                recipe_path_to_use = path
                break

        if not recipe_path_to_use:
            raise FileNotFoundError(f"Recipe file not found in any search path: {normalized_filename}")

        with open(recipe_path_to_use, 'r', encoding='utf-8') as f:
            content = f.read()

        for key, value in dynamic_values.items():
            if value is not None:
                content = content.replace(f"{{{{ {key} }}}}", str(value))
        
        main_recipe = yaml.safe_load(content)
        
        merged_recipe = { 'nodes': {}, 'connections': [], 'ui_map': {} }
        for key in self.injector_order:
             if key.startswith('dynamic_'):
                merged_recipe[key] = {}
        
        parent_recipe_dir = os.path.dirname(recipe_path_to_use)
        for import_path_template in main_recipe.get('imports', []):
            import_path = import_path_template
            for key, value in dynamic_values.items():
                if value is not None:
                    import_path = import_path.replace(f"{{{{ {key} }}}}", str(value))
            try:
                imported_recipe = self._load_and_merge_recipe(import_path, dynamic_values, search_context_dir=parent_recipe_dir)
                for key in merged_recipe:
                    if key == 'nodes' or key.startswith('dynamic_'):
                        merged_recipe[key].update(imported_recipe.get(key, {}))
                    elif key == 'connections':
                        merged_recipe[key].extend(imported_recipe.get(key, []))
                    elif key == 'ui_map':
                        merged_recipe[key].update(imported_recipe.get(key, {}))
            except FileNotFoundError:
                print(f"Warning: Optional recipe partial '{import_path}' not found. Skipping.")

        for key in list(merged_recipe.keys()) + list(main_recipe.keys()):
             if key not in merged_recipe and not key.startswith('dynamic_'): continue
             if key == 'nodes' or key.startswith('dynamic_'):
                 merged_recipe.setdefault(key, {}).update(main_recipe.get(key, {}))
             elif key == 'connections':
                 merged_recipe[key].extend(main_recipe.get(key, []))
             elif key == 'ui_map':
                 merged_recipe[key].update(main_recipe.get(key, {}))
        
        return merged_recipe

    def _get_injector_function(self, chain_type):
        if chain_type in self.loaded_local_injectors:
            return self.loaded_local_injectors[chain_type]

        if self.base_path:
            injector_module_name = chain_type.replace('dynamic_', '').replace('_chains', '_injector')
            injector_file_path = os.path.join(self.base_path, f"{injector_module_name}.py")
            
            if os.path.exists(injector_file_path):
                try:
                    spec = importlib.util.spec_from_file_location(injector_module_name, injector_file_path)
                    module = importlib.util.module_from_spec(spec)
                    
                    original_sys_path = sys.path[:]
                    if self.base_path not in sys.path:
                        sys.path.insert(0, self.base_path)
                    
                    spec.loader.exec_module(module)
                    
                    sys.path[:] = original_sys_path

                    if hasattr(module, 'inject'):
                        print(f"Dynamically loaded local injector: {injector_file_path}")
                        self.loaded_local_injectors[chain_type] = module.inject
                        return module.inject
                except Exception as e:
                    print(f"Error loading local injector {injector_file_path}: {e}")
        
        if chain_type in self.global_injectors:
            return self.global_injectors[chain_type]
            
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