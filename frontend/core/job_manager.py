import uuid
import time
import threading
import json
from copy import deepcopy
from typing import Dict, Any, List, Optional
import gradio as gr

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

def get_latest_running_job_for_module(module_name: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        latest_job = None
        latest_time = 0
        for job in _jobs.values():
            if (job.get('module') and job['module'].__name__ == module_name and
                    job['status'] in [STATUS_QUEUED, STATUS_PROCESSING]):
                if job['updated_at'] > latest_time:
                    latest_time = job['updated_at']
                    latest_job = job
        
        if latest_job:
            job_copy = latest_job.copy()
            job_copy.pop('module', None)
            return job_copy
        return None


def create_job(ui_values: Dict[str, Any], module: Any) -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": STATUS_QUEUED,
            "progress_message": "Status: Queued...",
            "result_files": None,
            "error_message": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "ui_values": ui_values, 
            "module": module 
        }
    print(f"[JobManager] Created job {job_id}")
    return job_id

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        return _jobs.get(job_id, {}).copy()

def update_job(job_id: str, status: str, progress_message: str = "", result_files: Optional[List[str]] = None, error_message: Optional[str] = None):
    with _jobs_lock:
        if job_id in _jobs:
            job = _jobs[job_id]
            job["status"] = status
            if progress_message:
                job["progress_message"] = progress_message
            if result_files is not None: 
                job["result_files"] = result_files
            if error_message:
                job["error_message"] = error_message
            job["updated_at"] = time.time()
            print(f"[JobManager] Updated job {job_id}: Status={status}, Message='{progress_message or error_message}'")
    

def run_job_in_background(job_id: str):
    job_info = get_job(job_id)
    if not job_info:
        print(f"[JobManager] Error: Could not find job {job_id} to run.")
        return

    module = job_info["module"]
    ui_values = job_info["ui_values"]

    def worker():
        try:
            update_job(job_id, STATUS_PROCESSING, "Status: Starting generation...")
            
            final_files = []
            for updates in module.run_generation(ui_values):
                status_message = updates[0]
                
                potential_outputs = updates[1:]
                
                last_item = potential_outputs[-1] if potential_outputs else None
                if isinstance(last_item, dict) and last_item.get("__type__") == "update":
                    potential_outputs = potential_outputs[:-1]

                current_files = []
                for item in potential_outputs:
                    if item is None or (isinstance(item, dict) and item.get("__type__") == "update"):
                        continue
                    if isinstance(item, list):
                        current_files.extend(f for f in item if f is not None)
                    else: 
                        current_files.append(item)
                
                if current_files:
                    final_files = current_files
                
                update_job(job_id, STATUS_PROCESSING, progress_message=status_message, result_files=final_files)

            last_job_state = get_job(job_id)
            final_files_from_last_state = last_job_state.get('result_files', [])
            
            update_job(job_id, STATUS_COMPLETED, progress_message="Status: Loaded successfully!", result_files=final_files_from_last_state)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error: A critical error occurred: {e}"
            update_job(job_id, STATUS_FAILED, error_message=error_msg)

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()

def get_completed_jobs(limit: int = 100) -> List[Dict[str, Any]]:
    with _jobs_lock:
        completed = [job.copy() for job in _jobs.values() if job["status"] == STATUS_COMPLETED and job.get("result_files")]
    
    completed.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    
    return completed[:limit]