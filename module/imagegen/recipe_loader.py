import os
import yaml
from .shared.config_loader import load_features_config, deep_merge_dicts

IMAGE_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_RECIPE_DIR = os.path.join(IMAGE_GEN_DIR, "workflow_recipes")


def load_and_merge_recipe(recipe_filename: str, dynamic_values: dict = None, base_path: str = None, search_context_dir: str = None) -> dict:
    dynamic_values = dynamic_values or {}
    normalized_filename = os.path.normpath(recipe_filename)
    
    search_paths = []
    if search_context_dir:
        search_paths.append(os.path.join(search_context_dir, normalized_filename))
    if base_path:
        search_paths.append(os.path.join(base_path, normalized_filename))
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
    
    main_recipe = yaml.safe_load(content) or {}
    
    merged_recipe = {'nodes': {}, 'connections': [], 'ui_map': {}}
    
    parent_recipe_dir = os.path.dirname(recipe_path_to_use)
    for import_path_template in main_recipe.get('imports', []):
        import_path = import_path_template
        for key, value in dynamic_values.items():
            if value is not None:
                import_path = import_path.replace(f"{{{{ {key} }}}}", str(value))
        try:
            imported_recipe = load_and_merge_recipe(import_path, dynamic_values, base_path=base_path, search_context_dir=parent_recipe_dir)
            for key in imported_recipe:
                if key == 'nodes' or key.startswith('dynamic_'):
                    merged_recipe.setdefault(key, {}).update(imported_recipe.get(key, {}))
                elif key == 'connections':
                    merged_recipe.setdefault(key, []).extend(imported_recipe.get(key, []))
                elif key == 'ui_map':
                    merged_recipe.setdefault(key, {}).update(imported_recipe.get(key, {}))
        except FileNotFoundError:
            print(f"Warning: Optional recipe partial '{import_path}' not found. Skipping.")

    for key, value in main_recipe.items():
        if key == 'imports':
            continue
        if key == 'nodes' or key.startswith('dynamic_'):
            merged_recipe.setdefault(key, {}).update(value if isinstance(value, dict) else {})
        elif key == 'connections':
            merged_recipe.setdefault('connections', []).extend(value if isinstance(value, list) else [])
        elif key == 'ui_map':
            merged_recipe.setdefault('ui_map', {}).update(value if isinstance(value, dict) else {})
        else:
            merged_recipe[key] = value

    return merged_recipe


def get_injector_order(model_type: str = None) -> list:
    order = []
    try:
        features_config = load_features_config()

        if model_type and model_type in features_config:
            enabled_features = features_config[model_type].get('enabled_chains', [])
        elif 'default' in features_config:
            enabled_features = features_config['default'].get('enabled_chains', [])
        else:
            enabled_features = []

        for feat in enabled_features:
            chain_key = f"dynamic_{feat}_chains"
            if chain_key not in order:
                order.append(chain_key)

    except Exception as e:
        print(f"Warning: Could not load image_gen_features.yaml. Error: {e}")

    return order


def load_recipe_and_injector_order(recipe_filename: str, dynamic_values: dict = None, base_path: str = None):
    dynamic_values = dynamic_values or {}
    model_type = dynamic_values.get('model_type')
    
    recipe = load_and_merge_recipe(recipe_filename, dynamic_values, base_path=base_path)
    injector_order = get_injector_order(model_type=model_type)
    
    return recipe, injector_order
