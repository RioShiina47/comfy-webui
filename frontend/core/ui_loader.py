import os
import importlib
import sys
import yaml
from collections import defaultdict
from core.yaml_loader import load_and_merge_yaml

def load_ui_list():
    config = load_and_merge_yaml("ui_list.yaml")
    include_list = config.get("include", [])
    if include_list is None:
        return []
    return include_list

def discover_ui_modules(ui_list=None):
    ui_tree = defaultdict(list)
    ui_modules = {}
    
    if 'custom' not in sys.path:
        sys.path.insert(0, 'custom')

    ui_dirs_to_scan = ["module", "custom/module"]
    
    if not ui_list:
        print("UI include list is empty. Discovering all UI modules...")
        for ui_dir in ui_dirs_to_scan:
            if not os.path.isdir(ui_dir):
                continue
            for root, _, files in os.walk(ui_dir):
                for filename in files:
                    if filename.endswith("_ui.py") and filename != "__init__.py":
                        module_path_parts = os.path.normpath(os.path.join(root, filename[:-3])).split(os.sep)
                        module_name = ".".join(module_path_parts)
                        _load_and_register_module(module_name, ui_tree, ui_modules)
    else:
        print(f"Loading specified UI modules from ui_list.yaml: {ui_list}")
        for module_name in ui_list:
            _load_and_register_module(module_name, ui_tree, ui_modules)

    return ui_tree, ui_modules


def _load_and_register_module(module_name, ui_tree, ui_modules):
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "UI_INFO"):
            info = module.UI_INFO
            if "main_tab" not in info or "sub_tab" not in info:
                print(f"Skipping module {module_name}: UI_INFO missing 'main_tab' or 'sub_tab'.")
                return
            
            main_tab_name = info["main_tab"]
            sub_tab_name = info["sub_tab"]
            
            ui_tree[main_tab_name].append(info)
            ui_modules[sub_tab_name] = module
            print(f"Successfully loaded UI module: {module_name}")
    except ModuleNotFoundError:
        print(f"Error: UI module '{module_name}' specified in ui_list.yaml not found.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading UI module {module_name}: {e}")


def load_ui_layout():
    custom_layout_path = os.path.join("custom", "yaml", "ui_layout.yaml")
    base_layout_path = os.path.join("yaml", "ui_layout.yaml")
    
    layout_path_to_use = None
    if os.path.exists(custom_layout_path):
        print(f"Loading custom UI layout from: {custom_layout_path}")
        layout_path_to_use = custom_layout_path
    elif os.path.exists(base_layout_path):
        layout_path_to_use = base_layout_path
        
    layout_config = {"main_tabs_order": [], "sub_tabs_order": {}}
    
    if not layout_path_to_use:
        print(f"Warning: Could not find UI layout configuration. Falling back to alphabetical sorting.")
        return layout_config

    try:
        with open(layout_path_to_use, 'r', encoding='utf-8') as f:
            layout = yaml.safe_load(f)

        if isinstance(layout, dict) and "main_tabs_order" in layout:
            for item in layout["main_tabs_order"]:
                if isinstance(item, dict):
                    main_tab_name = next(iter(item))
                    layout_config["main_tabs_order"].append(main_tab_name)
                    layout_config["sub_tabs_order"][main_tab_name] = item[main_tab_name]
                else:
                    layout_config["main_tabs_order"].append(item)
            return layout_config
    except Exception as e:
        print(f"Warning: Could not load or parse UI layout configuration from '{layout_path_to_use}': {e}. Falling back to alphabetical sorting.")
    
    return layout_config