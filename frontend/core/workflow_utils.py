import datetime
import uuid

def find_node_by_title(workflow_data, title):
    for node_id, node_info in workflow_data.items():
        if node_info.get("_meta", {}).get("title") == title:
            return node_id
    return None

def set_node_param(workflow_data, node_id, param_name, value):
    if node_id in workflow_data and "inputs" in workflow_data[node_id]:
        workflow_data[node_id]["inputs"][param_name] = value
        print(f"Updated node '{node_id}' ({workflow_data[node_id].get('_meta', {}).get('title', 'Untitled')}), param '{param_name}' to: {value}")
    else:
        print(f"Warning: Node '{node_id}' or its 'inputs' not found for param '{param_name}'.")
    return workflow_data

def find_output_node_id(workflow_data):
    for node_id, node_info in workflow_data.items():
        if node_info.get("class_type") == "SaveImage":
            print(f"Found output node (SaveImage) with ID: {node_id}")
            return node_id
    print("Warning: No 'SaveImage' node found in workflow.")
    return None

def get_filename_prefix():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S') + f"-{now.microsecond // 1000:03d}"