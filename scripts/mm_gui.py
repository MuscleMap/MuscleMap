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

def run_segmentation(segment_input_image_entry, segment_region_entry, segment_file_path_entry, segment_average):
    # Start with the mandatory arguments
    # Assuming mm_segment.py is in the same directory as your current script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_segment.py')
    
    command = [
        sys.executable, script_path,  # Use the full path to the mm_segment.py script
        '-i', segment_input_image_entry.get(),
        '-r', segment_region_entry.get(), 
        '-g', segment_average.get()
    ]

    # Add optional arguments only if they have values
    if segment_file_path_entry.get():
        command.extend(['-o', segment_file_path_entry.get()])

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




def run_gmm_kmeans_chained(chain_input_image_entry, chain_region_entry, chain_output_dir_entry, chain_method_var, chain_components_var, chain_use_gpu):
    input_image = chain_input_image_entry.get()
    region = chain_region_entry.get()
    output_dir = chain_output_dir_entry.get()
    method = chain_method_var.get()
    components = chain_components_var.get()
    use_gpu= chain_use_gpu.get()

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
        '-g', use_gpu,
        '-o', output_dir
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
        messagebox.showinfo("Success", "Segmentation and Extract Metrics completed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def run_dixon_chained(fat_entry, water_entry, region_entry, chain_output_dir_entry, chain_use_gpu):
    fat = fat_entry.get()
    water = water_entry.get()
    region= region_entry.get()
    output_dir = chain_output_dir_entry.get()
    use_gpu= chain_use_gpu.get()

    segmentation_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_segment.py')
    extraction_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_extract_metrics.py')

    if water.endswith('.nii.gz'):
        water_output = os.path.join(output_dir, os.path.basename(water).replace('.nii.gz', '_dseg.nii.gz'))
    elif water.endswith('.nii'):
        water_output = os.path.join(output_dir, os.path.basename(water).replace('.nii', '_dseg.nii.gz'))

    # Segmentation command
    water_command = [
        sys.executable, segmentation_script_path,
        '-i', water,
        '-r', region, 
        '-g', use_gpu,
        '-o', output_dir
    ]
    print("Segmentation command to be executed:", water_command)  # Debug print
    try:
        subprocess.run(water_command, check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

    fat_command = [
        sys.executable, segmentation_script_path,
        '-i', fat,
        '-r', region, 
        '-g', use_gpu,
        '-o', output_dir
    ]
    print("Segmentation command to be executed:", fat_command)  # Debug print

    try:
        subprocess.run(fat_command, check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")



    # Extraction command
    command = [
        sys.executable, extraction_script_path,
        '-m', 'dixon',
        '-r', region,
        '-o', output_dir,
        '-f', fat,
        '-w', water,
        '-s', water_output

    ]
    print("Extraction command to be executed:", command)  # Debug print

    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", "Segmentation and Extract Metrics completed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def run_average_chained(input_entry, region_entry, chain_output_dir_entry, chain_use_gpu):
    input = input_entry.get()
    region= region_entry.get()
    output_dir = chain_output_dir_entry.get()
    use_gpu= chain_use_gpu.get()

    segmentation_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_segment.py')
    extraction_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_extract_metrics.py')

    if input.endswith('.nii.gz'):
        output = os.path.join(output_dir, os.path.basename(input).replace('.nii.gz', '_dseg.nii.gz'))
    
    # Segmentation command
    command = [
        sys.executable, segmentation_script_path,
        '-i', input,
        '-r', region, 
        '-g', use_gpu,
        '-o', output_dir
    ]
    print("Segmentation command to be executed:", command)  # Debug print
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

    # Extraction command
    command = [
        sys.executable, extraction_script_path,
        '-m', 'average',
        '-r', region,
        '-o', output_dir,
        '-i', input,
        '-s', output

    ]
    print("Extraction command to be executed:", command)  # Debug print

    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", "Segmentation and Extract Metrics completed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def main():
    root = tk.Tk()
    root.title("MuscleMap Toolbox")

    # Tab Control
    tab_control = ttk.Notebook(root)
    segment_tab = ttk.Frame(tab_control)
    extract_tab = ttk.Frame(tab_control)
    chain_gmm_kmeans_tab = ttk.Frame(tab_control)
    chain_dixon_tab = ttk.Frame(tab_control)
    chain_average_tab = ttk.Frame(tab_control)
    tab_control.add(segment_tab, text='Segmentation')
    tab_control.add(extract_tab, text='Extract Metrics')
    tab_control.add(chain_gmm_kmeans_tab, text='Chaining GMM or Kmeans')
    tab_control.add(chain_dixon_tab, text='Chaining Dixon')
    tab_control.add(chain_average_tab, text='Chaining Average')
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
    segment_use_gpu = tk.StringVar()
    segments_frame = ttk.Frame(segment_tab)
    rb2 = ttk.Radiobutton(segments_frame, text="Yes", variable=segment_use_gpu, value='Y')
    rb3 = ttk.Radiobutton(segments_frame, text="No", variable=segment_use_gpu, value='N')
    rb2.pack(side='left')
    rb3.pack(side='left')
    segments_frame.grid(row=3, column=1)

    ttk.Button(segment_tab, text="Run Segmentation", command=lambda: run_segmentation(segment_input_image_entry, segment_region_entry, segment_file_path_entry, segment_use_gpu)).grid(row=4, column=1)

    # Extraction GUI
    ttk.Label(extract_tab, text="Method:").grid(row=0, column=0)
    extract_method_var = tk.StringVar(value='dixon')
    methods = ['dixon', 'kmeans', 'gmm', 'average']
    method_menu = ttk.OptionMenu(extract_tab, extract_method_var, extract_method_var.get(), *methods)

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
 

    # Gmm kmean chaining tab
    ttk.Label(chain_gmm_kmeans_tab, text="Segmentation and Extract Metrics").grid(row=0, column=0, columnspan=3)

    # Segmentation fields in the Chaining tab
    ttk.Label(chain_gmm_kmeans_tab, text="Input Image:").grid(row=1, column=0)
    chain_input_image_entry = tk.Entry(chain_gmm_kmeans_tab, width=50)
    chain_input_image_entry.grid(row=1, column=1)
    ttk.Button(chain_gmm_kmeans_tab, text="Browse...", command=lambda: browse_file(chain_input_image_entry)).grid(row=1, column=2)

    ttk.Label(chain_gmm_kmeans_tab, text="Region:").grid(row=2, column=0)
    chain_region_entry = tk.Entry(chain_gmm_kmeans_tab, width=50)
    chain_region_entry.grid(row=2, column=1)

    ttk.Label(chain_gmm_kmeans_tab, text="GPU:").grid(row=3, column=0)
    chain_use_gpu = tk.StringVar()
    chain_GPU_frame = ttk.Frame(chain_gmm_kmeans_tab)
    rb2 = ttk.Radiobutton(chain_GPU_frame, text="Yes", variable=chain_use_gpu, value='Y')
    rb3 = ttk.Radiobutton(chain_GPU_frame, text="No", variable=chain_use_gpu, value='N')
    rb2.pack(side='left')
    rb3.pack(side='left')
    chain_GPU_frame.grid(row=3, column=1)

    # Extraction fields in the Chaining tab
    ttk.Label(chain_gmm_kmeans_tab, text="Method:").grid(row=4, column=0)
    chain_method_var = tk.StringVar(value='kmeans')
    chaining_methods = ['kmeans', 'gmm']
    chain_method_menu = ttk.OptionMenu(chain_gmm_kmeans_tab, chain_method_var, chain_method_var.get(), *chaining_methods)
    chain_method_menu.grid(row=4, column=1)


    ttk.Label(chain_gmm_kmeans_tab, text="Components:").grid(row=5, column=0)
    chain_components_var = tk.StringVar()
    chain_components_frame = ttk.Frame(chain_gmm_kmeans_tab)
    chain_rb2 = ttk.Radiobutton(chain_components_frame, text="2", variable=chain_components_var, value='2')
    chain_rb3 = ttk.Radiobutton(chain_components_frame, text="3", variable=chain_components_var, value='3')
    chain_rb2.pack(side='left')
    chain_rb3.pack(side='left')
    chain_components_frame.grid(row=5, column=1)

    ttk.Label(chain_gmm_kmeans_tab, text="Output Directory:").grid(row=6, column=0)
    chain_output_dir_entry = tk.Entry(chain_gmm_kmeans_tab, width=50)
    chain_output_dir_entry.grid(row=6, column=1)
    ttk.Button(chain_gmm_kmeans_tab, text="Browse...", command=lambda: browse_directory(chain_output_dir_entry)).grid(row=6, column=2)
    ttk.Button(chain_gmm_kmeans_tab, text="Run", command=lambda: run_gmm_kmeans_chained(chain_input_image_entry, chain_region_entry, chain_output_dir_entry, chain_method_var, chain_components_var, chain_use_gpu)).grid(row=7, column=1)


    # Dixon Chaining GUI Elements
    ttk.Label(chain_dixon_tab, text="Segmentation and Extract Metrics").grid(row=0, column=0, columnspan=3)
    
    ttk.Label(chain_dixon_tab, text="Input Fat Image:").grid(row=1, column=0)
    dixon_fat_image_entry = tk.Entry(chain_dixon_tab, width=50)
    dixon_fat_image_entry.grid(row=1, column=1)
    ttk.Button(chain_dixon_tab, text="Browse...", command=lambda: browse_file(dixon_fat_image_entry)).grid(row=1, column=2)

    ttk.Label(chain_dixon_tab, text="Input Water Image:").grid(row=2, column=0)
    dixon_water_image_entry = tk.Entry(chain_dixon_tab, width=50)
    dixon_water_image_entry.grid(row=2, column=1)
    ttk.Button(chain_dixon_tab, text="Browse...", command=lambda: browse_file(dixon_water_image_entry)).grid(row=2, column=2)

    ttk.Label(chain_dixon_tab, text="Region:").grid(row=3, column=0)
    dixon_region_entry = tk.Entry(chain_dixon_tab, width=50)
    dixon_region_entry.grid(row=3, column=1)

    ttk.Label(chain_dixon_tab, text="Output Directory:").grid(row=4, column=0)
    dixon_output_dir_entry = tk.Entry(chain_dixon_tab, width=50)
    dixon_output_dir_entry.grid(row=4, column=1)
    ttk.Button(chain_dixon_tab, text="Browse...", command=lambda: browse_directory(dixon_output_dir_entry)).grid(row=4, column=2)

    ttk.Label(chain_dixon_tab, text="GPU:").grid(row=5, column=0)
    dixon_use_gpu = tk.StringVar(value='Y')
    dixon_frame = ttk.Frame(chain_dixon_tab)
    dixon_rb2 = ttk.Radiobutton(dixon_frame, text="Yes", variable=dixon_use_gpu, value='Y')
    dixon_rb3 = ttk.Radiobutton(dixon_frame, text="No", variable=dixon_use_gpu, value='N')
    dixon_rb2.pack(side='left')
    dixon_rb3.pack(side='left')
    dixon_frame.grid(row=5, column=1)

    # Button to execute the Dixon Chaining
    ttk.Button(chain_dixon_tab, text="Run", command=lambda: run_dixon_chained(dixon_fat_image_entry, dixon_water_image_entry, dixon_region_entry, dixon_output_dir_entry, dixon_use_gpu)).grid(row=6, column=1)
    
    # Average Chaining GUI Elements
    ttk.Label(chain_average_tab, text="Segmentation and Extract Metrics").grid(row=0, column=0, columnspan=3)

    ttk.Label(chain_average_tab, text="Input Image:").grid(row=1, column=0)
    average_image_entry = tk.Entry(chain_average_tab, width=50)
    average_image_entry.grid(row=1, column=1)
    ttk.Button(chain_average_tab, text="Browse...", command=lambda: browse_file(average_image_entry)).grid(row=1, column=2)

    ttk.Label(chain_average_tab, text="Region:").grid(row=2, column=0)
    average_region_entry = tk.Entry(chain_average_tab, width=50)
    average_region_entry.grid(row=2, column=1)

    ttk.Label(chain_average_tab, text="Output Directory:").grid(row=3, column=0)
    average_output_dir_entry = tk.Entry(chain_average_tab, width=50)
    average_output_dir_entry.grid(row=3, column=1)
    ttk.Button(chain_average_tab, text="Browse...", command=lambda: browse_directory(average_output_dir_entry)).grid(row=3, column=2)

    ttk.Label(chain_average_tab, text="GPU:").grid(row=4, column=0)
    average_use_gpu = tk.StringVar(value='Y')
    average_frame = ttk.Frame(chain_average_tab)
    average_rb2 = ttk.Radiobutton(average_frame, text="Yes", variable=average_use_gpu, value='Y')
    average_rb3 = ttk.Radiobutton(average_frame, text="No", variable=average_use_gpu, value='N')
    average_rb2.pack(side='left')
    average_rb3.pack(side='left')
    average_frame.grid(row=4, column=1)
    
    # Button to execute the average Chaining
    ttk.Button(chain_average_tab, text="Run", command=lambda: run_average_chained(average_image_entry, average_region_entry, average_output_dir_entry, average_use_gpu)).grid(row=5, column=1)
    

    root.mainloop()

if __name__ == "__main__":
    main()