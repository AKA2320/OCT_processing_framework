import sys
import multiprocessing
multiprocessing.freeze_support()
import logging
from registration_script import start_reg_gui

# =============================================================================
# Process Target Function & Stdout Redirector
# =============================================================================
class ProcessStdout:
    """A helper class to redirect stdout of a process to a multiprocessing.Queue."""
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass

def run_registration_process(output_queue, cancel_event, *args):
    """
    This function is the target for the multiprocessing.Process.
    It sets up stdout redirection and calls the main processing function.
    
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # A simple format without timestamps or log levels
        stream=sys.stdout  # Direct logging output to the redirected stderr
    )
    
    sys.stdout = ProcessStdout(output_queue)
    sys.stderr = ProcessStdout(output_queue)

    
    # Unpack arguments for gui_input
    (DATA_LOAD_DIR, USE_MODEL_LATERAL_TRANSLATION, SAVE_DETECTIONS, DATA_SAVE_DIR, expected_cells, 
     expected_surfaces, batch_data_flag) = args
    
    args_dict = {'DATA_LOAD_DIR':DATA_LOAD_DIR, 'USE_MODEL_LATERAL_TRANSLATION': USE_MODEL_LATERAL_TRANSLATION,
                'DATA_SAVE_DIR':DATA_SAVE_DIR, 'EXPECTED_CELLS':expected_cells, 'EXPECTED_SURFACES':expected_surfaces,
                'BATCH_FLAG':batch_data_flag, 'SAVE_DETECTIONS': SAVE_DETECTIONS}
    try:
        gui_start_process = multiprocessing.Process(target = start_reg_gui, kwargs=args_dict)
        gui_start_process.start()
        while gui_start_process.is_alive():
            try:
                if cancel_event.is_set():
                    gui_start_process.terminate()
                    break
            except Exception: # queue.Empty
                continue
        gui_start_process.join() 
    except Exception as e:
        # Ensure exceptions in the child process are reported to the GUI
        output_queue.put(f"\n--- A CRITICAL ERROR OCCURRED IN THE PROCESS ---\n")
        output_queue.put(str(e))
    finally:
        # Signal that the process is done
        if cancel_event.is_set():
            output_queue.put("PROCESS_CANCELLED")
        else:
            output_queue.put("PROCESS_FINISHED")
