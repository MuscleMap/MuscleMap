import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys

def browse_file(entry):
    filepath = filedialog.askopenfilename()
    if filepath:
        entry.delete(0, tk.END)
        entry.insert(0, filepath)

def browse_directory(entry):
    directory = filedialog.askdirectory()
    if directory:
        entry.delete(0, tk.END)
        entry.insert(0, directory)

def update_fields():
    method = method_var.get()
    if method == 'dixon':
        e_input_image.config(state='disabled')
        e_fat_image.config(state='normal')
        e_water_image.config(state='normal')
        rb2.config(state='disabled')
        rb3.config(state='disabled')
    elif method in ['kmeans', 'gmm']:
        e_input_image.config(state='normal')
        e_fat_image.config(state='disabled')
        e_water_image.config(state='disabled')
        rb2.config(state='normal')
        rb3.config(state='normal')

def run_segmentation():
    command = [
        sys.executable, 'mm_extract_metrics.py',
        '-m', method_var.get(),
        '-i', e_input_image.get(),
        '-f', e_fat_image.get(),
        '-w', e_water_image.get(),
        '-s', e_segmentation_image.get(),
        '-c', components_var.get(),
        '-r', e_region.get(),
        '-o', e_output_dir.get()
    ]
    command = [arg for arg in command if arg]  # Filter empty strings
    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", "Segmentation completed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("Muscle and Fat Segmentation Tool")

tk.Label(root, text="Method:").grid(row=0, column=0)
method_var = tk.StringVar(value='dixon')
methods = tk.OptionMenu(root, method_var, 'dixon', 'kmeans', 'gmm', command=lambda _: update_fields())
methods.grid(row=0, column=1)

tk.Label(root, text="Input Image:").grid(row=1, column=0)
e_input_image = tk.Entry(root, width=50, state='disabled')
e_input_image.grid(row=1, column=1)
tk.Button(root, text="Browse...", command=lambda: browse_file(e_input_image)).grid(row=1, column=2)

tk.Label(root, text="Fat Image:").grid(row=2, column=0)
e_fat_image = tk.Entry(root, width=50)
e_fat_image.grid(row=2, column=1)
tk.Button(root, text="Browse...", command=lambda: browse_file(e_fat_image)).grid(row=2, column=2)

tk.Label(root, text="Water Image:").grid(row=3, column=0)
e_water_image = tk.Entry(root, width=50)
e_water_image.grid(row=3, column=1)
tk.Button(root, text="Browse...", command=lambda: browse_file(e_water_image)).grid(row=3, column=2)

tk.Label(root, text="Segmentation Image:").grid(row=4, column=0)
e_segmentation_image = tk.Entry(root, width=50)
e_segmentation_image.grid(row=4, column=1)
tk.Button(root, text="Browse...", command=lambda: browse_file(e_segmentation_image)).grid(row=4, column=2)

tk.Label(root, text="Components:").grid(row=5, column=0)
components_var = tk.StringVar()
rb_frame = tk.Frame(root)
rb2 = tk.Radiobutton(rb_frame, text="2", variable=components_var, value='2', state='disabled')
rb3 = tk.Radiobutton(rb_frame, text="3", variable=components_var, value='3', state='disabled')
rb2.pack(side='left')
rb3.pack(side='left')
rb_frame.grid(row=5, column=1)

tk.Label(root, text="Region:").grid(row=6, column=0)
e_region = tk.Entry(root, width=50)
e_region.grid(row=6, column=1)

tk.Label(root, text="Output Directory:").grid(row=7, column=0)
e_output_dir = tk.Entry(root, width=50)
e_output_dir.grid(row=7, column=1)
tk.Button(root, text="Browse...", command=lambda: browse_directory(e_output_dir)).grid(row=7, column=2)

run_button = tk.Button(root, text="Run Segmentation", command=run_segmentation)
run_button.grid(row=8, column=1)

update_fields()  # Initial update to set the correct state for entry fields
root.mainloop()
