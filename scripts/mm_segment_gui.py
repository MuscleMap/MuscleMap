import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess

def browse_file(entry):
    filenames = filedialog.askopenfilenames()
    entry.delete(0, tk.END)
    entry.insert(0, ','.join(filenames))

def browse_folder(entry):
    foldername = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, foldername)

def run_gui_script(image, region, model, output_file, output_dir):
    command = [
        "python", "mm_segment.py",
        "-i", image,
        "-r", region,
        "-m", model,
        "-o", output_file,
        "-s", output_dir,
        "-g", "N"  # Ensure the GUI is not triggered again
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        messagebox.showinfo("Success", "Processing completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def launch_gui():
    root = tk.Tk()
    root.title("MM Segment GUI")

    tk.Label(root, text="Image:").grid(row=0, column=0, sticky=tk.W)
    image_entry = tk.Entry(root, width=50)
    image_entry.grid(row=0, column=1, padx=10)
    tk.Button(root, text="Browse", command=lambda: browse_file(image_entry)).grid(row=0, column=2)

    tk.Label(root, text="Region:").grid(row=1, column=0, sticky=tk.W)
    region_entry = tk.Entry(root, width=50)
    region_entry.grid(row=1, column=1, padx=10)

    tk.Label(root, text="Model:").grid(row=2, column=0, sticky=tk.W)
    model_entry = tk.Entry(root, width=50)
    model_entry.grid(row=2, column=1, padx=10)
    tk.Button(root, text="Browse", command=lambda: browse_file(model_entry)).grid(row=2, column=2)

    tk.Label(root, text="Output File Name:").grid(row=3, column=0, sticky=tk.W)
    output_file_entry = tk.Entry(root, width=50)
    output_file_entry.grid(row=3, column=1, padx=10)

    tk.Label(root, text="Output Directory:").grid(row=4, column=0, sticky=tk.W)
    output_dir_entry = tk.Entry(root, width=50)
    output_dir_entry.grid(row=4, column=1, padx=10)
    tk.Button(root, text="Browse", command=lambda: browse_folder(output_dir_entry)).grid(row=4, column=2)

    tk.Button(root, text="Run", command=lambda: run_gui_script(
        image_entry.get(),
        region_entry.get(),
        model_entry.get(),
        output_file_entry.get(),
        output_dir_entry.get()
    )).grid(row=5, column=1, pady=20)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()
