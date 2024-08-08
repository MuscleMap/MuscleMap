import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys

def browse_file(entry):
    """Open a file dialog and update the entry with the selected file path."""
    filepath = filedialog.askopenfilename()
    if filepath:
        entry.delete(0, tk.END)
        entry.insert(0, filepath)

def browse_directory(entry):
    """Open a directory dialog and update the entry with the selected directory path."""
    directory = filedialog.askdirectory()
    if directory:
        entry.delete(0, tk.END)
        entry.insert(0, directory)

def run_segmentation():
    input_images = e_input_images.get()
    region = e_region.get()
    file_path = e_file_path.get()
    model = e_model.get()
    use_gpu = gpu_var.get()
    
    if not all([input_images, region, file_path]):
        messagebox.showerror("Error", "Please fill all required fields.")
        return

    # Construct the command to run the Python script
    command = [
        sys.executable, 'mm_segment.py',
        '-i', input_images,
        '-r', region,
        '-f', file_path,
        '-m', model,
        '-g', use_gpu
    ]
    
    # Execute the Python script
    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", "Segmentation completed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Set up the GUI
root = tk.Tk()
root.title("Segmentation Tool")

tk.Label(root, text="Input Images:").grid(row=0, column=0)
e_input_images = tk.Entry(root, width=50)
e_input_images.grid(row=0, column=1)
tk.Button(root, text="Browse...", command=lambda: browse_file(e_input_images)).grid(row=0, column=2)

tk.Label(root, text="Region:").grid(row=1, column=0)
e_region = tk.Entry(root, width=50)
e_region.grid(row=1, column=1)

tk.Label(root, text="File Path:").grid(row=2, column=0)
e_file_path = tk.Entry(root, width=50)
e_file_path.grid(row=2, column=1)
tk.Button(root, text="Browse...", command=lambda: browse_directory(e_file_path)).grid(row=2, column=2)

tk.Label(root, text="Model: (optional)").grid(row=3, column=0)
e_model = tk.Entry(root, width=50)
e_model.grid(row=3, column=1)
tk.Button(root, text="Browse...", command=lambda: browse_file(e_model)).grid(row=3, column=2)

tk.Label(root, text="Use GPU:").grid(row=4, column=0)
gpu_var = tk.StringVar(value='N')
tk.Radiobutton(root, text="Yes", variable=gpu_var, value='Y').grid(row=4, column=1)
tk.Radiobutton(root, text="No", variable=gpu_var, value='N').grid(row=4, column=2)

run_button = tk.Button(root, text="Run Segmentation", command=run_segmentation)
run_button.grid(row=5, column=1)

root.mainloop()
