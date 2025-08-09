# main_gui.py
import multiprocessing
import sys
import os
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QMessageBox,
    QLineEdit,
    QCheckBox,
    QTextEdit,
    QTabWidget,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
import napari
from utils.util_funcs import GUI_load_h5, GUI_load_dcm
from GUI_scripts.gui_registration_script import gui_input

# =============================================================================
# Registration Tab Widget
# =============================================================================
class RegistrationTab(QWidget):
    # Signal to update the main window's status bar
    update_status = Signal(str)

    def __init__(self, mode='single', parent=None):
        """
        Initializes the RegistrationTab.
        Args:
            mode (str): 'single' for individual file/dir registration,
                        'batch' for batch directory registration.
            parent (QWidget): The parent widget.
        """
        super().__init__(parent)
        self.mode = mode
        self.selected_register_path = None
        self.selected_save_path = "output/"  # Default save path
        self.registration_thread = None

        # --- Create Layout ---
        layout = QVBoxLayout(self)

        # --- UI Elements ---
        # Customize labels and tooltips based on the mode
        if self.mode == 'single':
            reg_label_text = "Select Data for Registration (.h5, .hdf5, .dcm):"
            browse_tooltip = "Select a .h5, .hdf5, or .dcm file for registration."
        else:  # batch mode
            reg_label_text = "Select Directory for Batch Registration (must contain .h5 files):"
            browse_tooltip = "Select a directory containing .h5 files for batch processing."

        # Input Path Selection
        layout.addWidget(QLabel(reg_label_text))
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setToolTip(browse_tooltip)
        layout.addWidget(self.browse_btn)

        self.path_display = QLineEdit(f"No {self.mode} data selected for registration")
        self.path_display.setReadOnly(True)
        self.path_display.setStyleSheet("background-color: white; color: black; font-style: italic;")
        layout.addWidget(self.path_display)

        # Output Path Selection
        layout.addWidget(QLabel("Select Directory for Saving Results (Default: 'output/'):"))
        self.browse_save_btn = QPushButton("Browse Save Directory...")
        layout.addWidget(self.browse_save_btn)

        self.save_path_display = QLineEdit(self.selected_save_path)
        self.save_path_display.setReadOnly(True)
        self.save_path_display.setStyleSheet("background-color: white; color: black; font-style: normal;")
        layout.addWidget(self.save_path_display)

        # Registration Parameters
        layout.addWidget(QLabel("Expected Cells (int, default: 2):"))
        self.expected_cells_input = QLineEdit("2")
        self.expected_cells_input.setValidator(QIntValidator())
        self.expected_cells_input.setStyleSheet("background-color: white; color: black;")
        layout.addWidget(self.expected_cells_input)

        layout.addWidget(QLabel("Expected Surfaces (int, default: 2):"))
        self.expected_surfaces_input = QLineEdit("2")
        self.expected_surfaces_input.setValidator(QIntValidator())
        self.expected_surfaces_input.setStyleSheet("background-color: white; color: black;")
        layout.addWidget(self.expected_surfaces_input)

        self.use_model_x_checkbox = QCheckBox("USE_MODEL_X")
        self.use_model_x_checkbox.setChecked(True)
        layout.addWidget(self.use_model_x_checkbox)

        self.disable_tqdm_checkbox = QCheckBox("DISABLE_TQDM")
        self.disable_tqdm_checkbox.setChecked(True)
        layout.addWidget(self.disable_tqdm_checkbox)

        # Action Buttons
        self.register_btn = QPushButton("Register Data")
        self.register_btn.setToolTip("Runs the registration script on the selected data.")
        self.register_btn.setEnabled(False)
        self.register_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.register_btn)

        self.cancel_btn = QPushButton("Cancel Registration")
        self.cancel_btn.setToolTip("Terminates the ongoing registration script.")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.cancel_btn)

        # Output Log
        layout.addWidget(QLabel("--- Script Output ---"))
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setPlaceholderText("Script output will appear here...")
        self.output_log.setStyleSheet("background-color: white; color: black; border: 1px solid #ccc; padding: 5px;")
        layout.addWidget(self.output_log)
        layout.addStretch()

        # --- Connect Signals to Slots ---
        self.browse_btn.clicked.connect(self.select_registration_path)
        self.browse_save_btn.clicked.connect(self.select_save_directory)
        self.register_btn.clicked.connect(self.start_registration)
        self.cancel_btn.clicked.connect(self.cancel_registration)

    def _update_register_button_state(self):
        """Enables or disables the register button based on path selections."""
        if self.selected_register_path and self.selected_save_path:
            self.register_btn.setEnabled(True)
        else:
            self.register_btn.setEnabled(False)

    def select_registration_path(self):
        """Opens a file or directory dialog based on the tab's mode."""
        path = ""
        if self.mode == 'single':
            path, _ = QFileDialog.getOpenFileName(self, "Select Data for Registration", "", "Supported Files (*.h5 *.hdf5 *.dcm);;All Files (*)")
            if path and path.lower().endswith('.dcm'):
                path = os.path.dirname(path) # Use the directory if a .dcm file is selected
        else: # batch mode
            path = QFileDialog.getExistingDirectory(self, "Select Batch Directory for Registration")

        if path:
            self.selected_register_path = path
            self.path_display.setText(path)
            self.path_display.setStyleSheet("background-color: white; color: black; font-style: normal;")
            self.output_log.clear()
        else:
            self.selected_register_path = None
            self.path_display.setText(f"No {self.mode} data selected for registration")
            self.path_display.setStyleSheet("background-color: white; color: black; font-style: italic;")

        self._update_register_button_state()

    def select_save_directory(self):
        """Opens a dialog to select the save directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Directory for Saving Results")
        if path:
            self.selected_save_path = path
        else:
            self.selected_save_path = "output/" # Revert to default if dialog is cancelled

        self.save_path_display.setText(self.selected_save_path)
        self._update_register_button_state()

    def start_registration(self):
        """Validates inputs and starts the registration thread."""
        if not self.selected_register_path or not self.selected_save_path:
            QMessageBox.warning(self, "Warning", "Both an input and an output path must be selected.")
            return

        try:
            expected_cells = int(self.expected_cells_input.text())
            expected_surfaces = int(self.expected_surfaces_input.text())
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Expected Cells and Expected Surfaces must be valid integers.")
            return

        self.output_log.clear()
        self.append_output(f"Starting {self.mode} registration for: {os.path.basename(self.selected_register_path)}...")
        self.update_status.emit(f"Registration process initiated for: {os.path.basename(self.selected_register_path)}...")

        # --- UI State Update ---
        self.register_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        # --- Create and Start Thread ---
        self.registration_thread = RegistrationThread(
            directory_path=self.selected_register_path,
            save_directory_path=self.selected_save_path,
            use_model_x=self.use_model_x_checkbox.isChecked(),
            disable_tqdm=self.disable_tqdm_checkbox.isChecked(),
            expected_cells=expected_cells,
            expected_surfaces=expected_surfaces,
            batch_data_flag=(self.mode == 'batch')
        )
        # Connect thread signals to this tab's slots
        self.registration_thread.output_ready.connect(self.append_output)
        self.registration_thread.error_ready.connect(self.append_output)
        self.registration_thread.finished.connect(self.process_finished)
        self.registration_thread.registration_cancelled.connect(self.process_cancelled)
        self.registration_thread.start()

    def cancel_registration(self):
        """Requests cancellation of the running registration thread."""
        if self.registration_thread and self.registration_thread.isRunning():
            self.update_status.emit("Cancelling registration process...")
            self.append_output("User requested cancellation. Terminating process...")
            self.registration_thread.terminate_process()

    def process_finished(self):
        """Handles the completion of the registration process."""
        self.update_status.emit("Registration process finished.")
        self.append_output("Registration process finished successfully.")
        self._reset_ui_state()

    def process_cancelled(self):
        """Handles the UI update after a process has been cancelled."""
        self.update_status.emit("Registration process cancelled.")
        self.append_output("Process successfully cancelled.")
        self._reset_ui_state()

    def _reset_ui_state(self):
        """Resets the UI controls to their default state after a process ends."""
        self.register_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.registration_thread = None

    def append_output(self, text):
        """Appends text to the output log and scrolls to the bottom."""
        # Use moveCursor and insertPlainText to avoid adding extra newlines
        cursor = self.output_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.output_log.setTextCursor(cursor)
        self.output_log.insertPlainText(text)


# =============================================================================
# Stdout Redirector
# =============================================================================
class StdoutRedirector(QObject):
    """
    A QObject that redirects stdout to a signal.
    This is used to capture print statements from a function
    running in a separate thread and display them in the GUI.
    """
    text_written = Signal(str)

    def write(self, text):
        """
        This method is called by `print`. It emits a signal with the text.
        """
        self.text_written.emit(text)

    def flush(self):
        """
        This method is needed for the file-like object interface.
        """
        pass


# =============================================================================
# Registration Worker Thread
# =============================================================================
class RegistrationThread(QThread):
    """
    Worker thread to run the registration script without blocking the GUI.
    """
    output_ready = Signal(str)
    error_ready = Signal(str)
    finished = Signal()
    registration_cancelled = Signal()

    def __init__(self, directory_path, save_directory_path, use_model_x, disable_tqdm, expected_cells, expected_surfaces, batch_data_flag):
        super().__init__()
        self.directory_path = directory_path
        self.save_directory_path = save_directory_path
        self.use_model_x = use_model_x
        self.disable_tqdm = disable_tqdm
        self.expected_cells = expected_cells
        self.expected_surfaces = expected_surfaces
        self.batch_data_flag = batch_data_flag
        self._cancellation_flag = False

    def run(self):
        """Executes the registration process and redirects stdout."""
        # --- Redirect stdout to capture print statements ---
        stdout_redirector = StdoutRedirector()
        stdout_redirector.text_written.connect(self.output_ready)
        original_stdout = sys.stdout
        sys.stdout = stdout_redirector

        try:
            # The external script is called here.
            # Any print() statements inside gui_input will now be captured.
            gui_input(
                dirname=self.directory_path,
                use_model_x=self.use_model_x,
                disable_tqdm=self.disable_tqdm,
                save_dirname=self.save_directory_path,
                expected_cells=self.expected_cells,
                expected_surfaces=self.expected_surfaces,
                batch_data_flag=self.batch_data_flag,
                cancellation_flag=lambda: self._cancellation_flag
            )
            # Check flag after execution to see if it was a cancellation
            if self._cancellation_flag:
                self.registration_cancelled.emit()
            else:
                self.finished.emit()
        except Exception as e:
            self.error_ready.emit(f"\nAn error occurred in the registration script: {e}\n")
            # Also emit finished to ensure the UI resets
            self.finished.emit()
        finally:
            # --- IMPORTANT: Restore original stdout ---
            sys.stdout = original_stdout

    def terminate_process(self):
        """Sets the flag to signal the running process to terminate."""
        self._cancellation_flag = True


# =============================================================================
# Main Application Window
# =============================================================================
class PathLoaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Loader & Processor")
        self.resize(600, 600)

        self.selected_load_path = None

        # --- Main Layout and Tab Widget ---
        overall_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        overall_layout.addWidget(self.tab_widget)

        # --- Tab 1: Data Loading & Visualization ---
        self.create_load_tab()

        # --- Tab 2: Data Registration (Single) ---
        self.single_register_tab = RegistrationTab(mode='single')
        self.single_register_tab.update_status.connect(self.update_status_bar)
        self.tab_widget.addTab(self.single_register_tab, "Register Data")

        # --- Tab 3: Batch Data Registration ---
        self.batch_register_tab = RegistrationTab(mode='batch')
        self.batch_register_tab.update_status.connect(self.update_status_bar)
        self.tab_widget.addTab(self.batch_register_tab, "Batch Register Data")

    def create_load_tab(self):
        """Creates the 'Load & Visualize' tab."""
        self.load_tab_widget = QWidget()
        load_layout = QVBoxLayout(self.load_tab_widget)

        self.browse_load_btn = QPushButton("Browse File/Directory...")
        self.browse_load_btn.setToolTip("Select an HDF5 file or any file within a DICOM directory.")
        load_layout.addWidget(self.browse_load_btn)

        self.path_display_load = QLineEdit("No file/directory selected for loading")
        self.path_display_load.setReadOnly(True)
        self.path_display_load.setStyleSheet("background-color: white; color: black; font-style: italic;")
        load_layout.addWidget(self.path_display_load)

        self.visualize_checkbox = QCheckBox("Visualize with napari")
        self.visualize_checkbox.setChecked(True)
        load_layout.addWidget(self.visualize_checkbox)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.setEnabled(False)
        self.load_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        load_layout.addWidget(self.load_btn)

        self.status_label_load = QLabel("Please use 'Browse File/Directory...' to select data.")
        self.status_label_load.setStyleSheet("padding: 5px;")
        load_layout.addWidget(self.status_label_load)
        load_layout.addStretch()

        self.browse_load_btn.clicked.connect(self.select_load_path)
        self.load_btn.clicked.connect(self.process_load_path)

        self.tab_widget.addTab(self.load_tab_widget, "Load & Visualize")

    def select_load_path(self):
        """Opens a dialog to select a file or directory for loading."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File or a File in a DICOM Directory",
            "",
            "Supported Files (*.h5 *.hdf5 *.dcm);;All Files (*)"
        )
        if not file_path:
            return

        # Determine if it's a file or a directory (from a .dcm file)
        if file_path.lower().endswith(('.h5', '.hdf5')):
            path_to_use = file_path
        elif file_path.lower().endswith('.dcm'):
            path_to_use = os.path.dirname(file_path)
        else:
            QMessageBox.warning(self, "Unsupported File", "Please select a .h5, .hdf5, or .dcm file.")
            self.load_btn.setEnabled(False)
            return

        self.selected_load_path = path_to_use
        self.path_display_load.setText(path_to_use)
        self.path_display_load.setStyleSheet("background-color: white; color: black; font-style: normal;")
        self.update_status_bar("Path selected. Ready to load.")
        self.load_btn.setEnabled(True)

    def process_load_path(self):
        """Loads data from the selected path and optionally visualizes it."""
        path = self.selected_load_path
        if not path:
            QMessageBox.warning(self, "Warning", "No path has been selected for loading.")
            return

        data = None
        try:
            if os.path.isdir(path):
                self.update_status_bar(f"Loading DICOM directory: {os.path.basename(path)}...")
                data = GUI_load_dcm(path)
            elif os.path.isfile(path) and path.lower().endswith(('.h5', '.hdf5')):
                self.update_status_bar(f"Loading H5 file: {os.path.basename(path)}...")
                data = GUI_load_h5(path)
            else:
                QMessageBox.critical(self, "Invalid Path", "The selected path is not a valid directory or HDF5 file.")
                return

            self.update_status_bar(f"Data loaded (Shape: {data.shape}).")

            if data is not None and self.visualize_checkbox.isChecked():
                self.update_status_bar(f"Visualizing with napari...")
                viewer = napari.view_image(data)
                self.update_status_bar(f"Visualization complete. Data Shape: {data.shape}")

        except Exception as e:
            self.update_status_bar("An error occurred during loading.")
            QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{e}")

    def update_status_bar(self, message):
        """Updates the status label on the Load tab."""
        self.status_label_load.setText(message)

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # This is necessary for running executables created by tools like PyInstaller
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = PathLoaderApp()
    window.show()
    sys.exit(app.exec())
