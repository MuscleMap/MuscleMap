import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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

def run_segmentation(segment_input_image_entry, segment_region_entry, segment_file_path_entry, segment_use_gui):
    # Start with the mandatory arguments
    # Assuming mm_segment.py is in the same directory as your current script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_segment.py')
    
    command = [
        sys.executable, script_path,  # Use the full path to the mm_segment.py script
        '-i', segment_input_image_entry.get(),
        '-r', segment_region_entry.get(), 
        '-g', segment_use_gui.get()
    ]

    # Add optional arguments only if they have values
    if segment_file_path_entry.get():
        command.extend(['-f', segment_file_path_entry.get()])

    print("Segmentation command to be executed:", command)  # Debug print

    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", "Segmentation completed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def run_extraction(extract_method_var, extract_region_entry, extract_output_dir_entry, extract_segmentation_image_entry, extract_fat_image_entry, extract_water_image_entry, extract_input_image_entry, extract_components_var, segmentation_output=None):
    method = extract_method_var.get()
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_extract_metrics.py')

    # Base command using the full path to the script
    command = [
        sys.executable, script_path,
        '-m', method,
        '-r', extract_region_entry.get(),
        '-o', extract_output_dir_entry.get()
    ]

    # Always add segmentation image if provided
    if segmentation_output or extract_segmentation_image_entry.get():
        command.extend(['-s', segmentation_output or extract_segmentation_image_entry.get()])

    # Add method-specific arguments only if they have values
    if method == 'dixon':
        if extract_fat_image_entry.get():
            command.extend(['-f', extract_fat_image_entry.get()])
        if extract_water_image_entry.get():
            command.extend(['-w', extract_water_image_entry.get()])
    else:
        if extract_input_image_entry.get():
            command.extend(['-i', extract_input_image_entry.get()])
        if extract_components_var.get():
            command.extend(['-c', extract_components_var.get()])

    print("Extraction command to be executed:", command)  # Debug print

    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", "Extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")




def run_chained(chain_input_image_entry, chain_region_entry, chain_output_dir_entry, chain_method_var, chain_components_var, chain_use_gui):
    input_image = chain_input_image_entry.get()
    region = chain_region_entry.get()
    output_dir = chain_output_dir_entry.get()
    method = chain_method_var.get()
    components = chain_components_var.get()
    use_gui= chain_use_gui.get()

    segmentation_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_segment.py')
    extraction_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_extract_metrics.py')

    if input_image.endswith('.nii.gz'):
        segmentation_output = os.path.join(output_dir, os.path.basename(input_image).replace('.nii.gz', '_dseg.nii.gz'))
    elif input_image.endswith('.nii'):
        segmentation_output = os.path.join(output_dir, os.path.basename(input_image).replace('.nii', '_dseg.nii.gz'))

    # Segmentation command
    command = [
        sys.executable, segmentation_script_path,
        '-i', input_image,
        '-r', region, 
        '-g', use_gui,
        '-f', segmentation_output
    ]
    print("Segmentation command to be executed:", command)  # Debug print

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

    # Extraction command
    command = [
        sys.executable, extraction_script_path,
        '-m', method,
        '-r', region,
        '-o', output_dir,
        '-i', input_image,
        '-c', components, 
        '-s', segmentation_output

    ]
    print("Extraction command to be executed:", command)  # Debug print

    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", "Segmentation and Extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def recreate_method_menu(method_menu, extract_method_var, methods):
    method_menu['menu'].delete(0, 'end')
    for method in methods:
        method_menu['menu'].add_command(label=method, command=tk._setit(extract_method_var, method, update_extract_fields))

def update_extract_fields(methods, extract_method_var, method_menu, extract_input_image_entry, input_image_browse_button, extract_fat_image_entry, fat_image_browse_button, extract_water_image_entry, water_image_browse_button, rb2, rb3, _=None):
    method = extract_method_var.get()
    recreate_method_menu(method_menu, extract_method_var, methods)  # Ensure the menu is updated correctly every time
    if method == 'dixon':
        extract_input_image_entry.config(state='disabled')
        input_image_browse_button.config(state='disabled')
        
        extract_fat_image_entry.config(state='normal')
        fat_image_browse_button.config(state='normal')
        
        extract_water_image_entry.config(state='normal')
        water_image_browse_button.config(state='normal')
        
        rb2.config(state='disabled')
        rb3.config(state='disabled')
    else:
        extract_input_image_entry.config(state='normal')
        input_image_browse_button.config(state='normal')
        
        extract_fat_image_entry.config(state='disabled')
        fat_image_browse_button.config(state='disabled')
        
        extract_water_image_entry.config(state='disabled')
        water_image_browse_button.config(state='disabled')
        
        rb2.config(state='normal')
        rb3.config(state='normal')

def main():
    root = tk.Tk()
    root.title("Muscle Imaging Analysis Tool")

    # Tab Control
    tab_control = ttk.Notebook(root)
    segment_tab = ttk.Frame(tab_control)
    extract_tab = ttk.Frame(tab_control)
    chain_tab = ttk.Frame(tab_control)
    tab_control.add(segment_tab, text='Segmentation')
    tab_control.add(extract_tab, text='Extract Metrics')
    tab_control.add(chain_tab, text='Chaining')
    tab_control.pack(expand=1, fill="both")

    # Segmentation GUI
    ttk.Label(segment_tab, text="Input Image:").grid(row=0, column=0)
    segment_input_image_entry = tk.Entry(segment_tab, width=50)
    segment_input_image_entry.grid(row=0, column=1)
    ttk.Button(segment_tab, text="Browse...", command=lambda: browse_file(segment_input_image_entry)).grid(row=0, column=2)

    ttk.Label(segment_tab, text="Region:").grid(row=1, column=0)
    segment_region_entry = tk.Entry(segment_tab, width=50)
    segment_region_entry.grid(row=1, column=1)

    ttk.Label(segment_tab, text="Output File Path:").grid(row=2, column=0)
    segment_file_path_entry = tk.Entry(segment_tab, width=50)
    segment_file_path_entry.grid(row=2, column=1)
    ttk.Button(segment_tab, text="Browse...", command=lambda: browse_directory(segment_file_path_entry)).grid(row=2, column=2)

    ttk.Label(segment_tab, text="GPU:").grid(row=3, column=0)
    segment_use_gui = tk.StringVar()
    segments_frame = ttk.Frame(segment_tab)
    rb2 = ttk.Radiobutton(segments_frame, text="Yes", variable=segment_use_gui, value='Y')
    rb3 = ttk.Radiobutton(segments_frame, text="No", variable=segment_use_gui, value='N')
    rb2.pack(side='left')
    rb3.pack(side='left')
    segments_frame.grid(row=3, column=1)

    ttk.Button(segment_tab, text="Run Segmentation", command=lambda: run_segmentation(segment_input_image_entry, segment_region_entry, segment_file_path_entry, segment_use_gui)).grid(row=4, column=1)

    # Extraction GUI
    ttk.Label(extract_tab, text="Method:").grid(row=0, column=0)
    extract_method_var = tk.StringVar(value='dixon')
    methods = ['dixon', 'kmeans', 'gmm']
    method_menu = ttk.OptionMenu(
        extract_tab, extract_method_var, extract_method_var.get(), *methods,
        command=lambda value: update_extract_fields(methods, 
            extract_method_var=extract_method_var,
            method_menu=method_menu,
            extract_input_image_entry=extract_input_image_entry,
            input_image_browse_button=input_image_browse_button,
            extract_fat_image_entry=extract_fat_image_entry,
            fat_image_browse_button=fat_image_browse_button,
            extract_water_image_entry=extract_water_image_entry,
            water_image_browse_button=water_image_browse_button,
            rb2=rb2,
            rb3=rb3,
            _=value  # Passing the newly selected value from the OptionMenu
        )
    )    
    method_menu.grid(row=0, column=1)

    ttk.Label(extract_tab, text="Input Image:").grid(row=1, column=0)
    extract_input_image_entry = tk.Entry(extract_tab, width=50)
    extract_input_image_entry.grid(row=1, column=1)
    input_image_browse_button = ttk.Button(extract_tab, text="Browse...", command=lambda: browse_file(extract_input_image_entry))
    input_image_browse_button.grid(row=1, column=2)

    ttk.Label(extract_tab, text="Fat Image:").grid(row=2, column=0)
    extract_fat_image_entry = tk.Entry(extract_tab, width=50)
    extract_fat_image_entry.grid(row=2, column=1)
    fat_image_browse_button = ttk.Button(extract_tab, text="Browse...", command=lambda: browse_file(extract_fat_image_entry))
    fat_image_browse_button.grid(row=2, column=2)

    ttk.Label(extract_tab, text="Water Image:").grid(row=3, column=0)
    extract_water_image_entry = tk.Entry(extract_tab, width=50)
    extract_water_image_entry.grid(row=3, column=1)
    water_image_browse_button = ttk.Button(extract_tab, text="Browse...", command=lambda: browse_file(extract_water_image_entry))
    water_image_browse_button.grid(row=3, column=2)

    ttk.Label(extract_tab, text="Segmentation Image:").grid(row=4, column=0)
    extract_segmentation_image_entry = tk.Entry(extract_tab, width=50)
    extract_segmentation_image_entry.grid(row=4, column=1)
    ttk.Button(extract_tab, text="Browse...", command=lambda: browse_file(extract_segmentation_image_entry)).grid(row=4, column=2)

    ttk.Label(extract_tab, text="Components:").grid(row=5, column=0)
    extract_components_var = tk.StringVar()
    components_frame = ttk.Frame(extract_tab)
    rb2 = ttk.Radiobutton(components_frame, text="2", variable=extract_components_var, value='2')
    rb3 = ttk.Radiobutton(components_frame, text="3", variable=extract_components_var, value='3')
    rb2.pack(side='left')
    rb3.pack(side='left')
    components_frame.grid(row=5, column=1)

    ttk.Label(extract_tab, text="Region: (optional)").grid(row=6, column=0)
    extract_region_entry = tk.Entry(extract_tab, width=50)
    extract_region_entry.grid(row=6, column=1)

    ttk.Label(extract_tab, text="Output Directory:").grid(row=7, column=0)
    extract_output_dir_entry = tk.Entry(extract_tab, width=50)
    extract_output_dir_entry.grid(row=7, column=1)
    ttk.Button(extract_tab, text="Browse...", command=lambda: browse_directory(extract_output_dir_entry)).grid(row=7, column=2)

    ttk.Button(extract_tab, text="Run Extract Metrics", command=lambda: run_extraction(extract_method_var, extract_region_entry, extract_output_dir_entry, extract_segmentation_image_entry, extract_fat_image_entry, extract_water_image_entry, extract_input_image_entry, extract_components_var)).grid(row=8, column=1)
    update_extract_fields(methods, 
            extract_method_var=extract_method_var,
            method_menu=method_menu,
            extract_input_image_entry=extract_input_image_entry,
            input_image_browse_button=input_image_browse_button,
            extract_fat_image_entry=extract_fat_image_entry,
            fat_image_browse_button=fat_image_browse_button,
            extract_water_image_entry=extract_water_image_entry,
            water_image_browse_button=water_image_browse_button,
            rb2=rb2,
            rb3=rb3 
        )

    # Chaining Tab - Full setup for both segmentation and extraction
    ttk.Label(chain_tab, text="Segmentation and Extraction Workflow").grid(row=0, column=0, columnspan=3)

    # Segmentation fields in the Chaining tab
    ttk.Label(chain_tab, text="Input Image:").grid(row=1, column=0)
    chain_input_image_entry = tk.Entry(chain_tab, width=50)
    chain_input_image_entry.grid(row=1, column=1)
    ttk.Button(chain_tab, text="Browse...", command=lambda: browse_file(chain_input_image_entry)).grid(row=1, column=2)

    ttk.Label(chain_tab, text="Region:").grid(row=2, column=0)
    chain_region_entry = tk.Entry(chain_tab, width=50)
    chain_region_entry.grid(row=2, column=1)

    ttk.Label(chain_tab, text="GPU:").grid(row=3, column=0)
    chain_use_gui = tk.StringVar()
    chain_GPU_frame = ttk.Frame(chain_tab)
    rb2 = ttk.Radiobutton(chain_GPU_frame, text="Yes", variable=chain_use_gui, value='Y')
    rb3 = ttk.Radiobutton(chain_GPU_frame, text="No", variable=chain_use_gui, value='N')
    rb2.pack(side='left')
    rb3.pack(side='left')
    chain_GPU_frame.grid(row=3, column=1)

    # Extraction fields in the Chaining tab
    ttk.Label(chain_tab, text="Method:").grid(row=4, column=0)
    chain_method_var = tk.StringVar(value='kmeans')
    chaining_methods = ['kmeans', 'gmm']
    chain_method_menu = ttk.OptionMenu(chain_tab, chain_method_var, chain_method_var.get(), *chaining_methods)
    chain_method_menu.grid(row=4, column=1)


    ttk.Label(chain_tab, text="Components:").grid(row=5, column=0)
    chain_components_var = tk.StringVar()
    chain_components_frame = ttk.Frame(chain_tab)
    chain_rb2 = ttk.Radiobutton(chain_components_frame, text="2", variable=chain_components_var, value='2')
    chain_rb3 = ttk.Radiobutton(chain_components_frame, text="3", variable=chain_components_var, value='3')
    chain_rb2.pack(side='left')
    chain_rb3.pack(side='left')
    chain_components_frame.grid(row=5, column=1)

    ttk.Label(chain_tab, text="Output Directory:").grid(row=6, column=0)
    chain_output_dir_entry = tk.Entry(chain_tab, width=50)
    chain_output_dir_entry.grid(row=6, column=1)
    ttk.Button(chain_tab, text="Browse...", command=lambda: browse_directory(chain_output_dir_entry)).grid(row=6, column=2)



    ttk.Button(chain_tab, text="Run Chained Workflow", command=lambda: run_chained(chain_input_image_entry, chain_region_entry, chain_output_dir_entry, chain_method_var, chain_components_var, chain_use_gui)).grid(row=7, column=1)

    root.mainloop()

if __name__ == "__main__":
    main()
