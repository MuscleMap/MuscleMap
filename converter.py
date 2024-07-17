import nibabel as nib
import os
import tkinter as tk
from tkinter import filedialog

def convert_nii_to_niigz(input_path, output_path=None):
    # Load the .nii file
    img = nib.load(input_path)
    
    # If no output path is provided, create one by appending .gz to the input path
    if output_path is None:
        output_path = input_path + '.gz'
    
    # Save the image as .nii.gz
    nib.save(img, output_path)
    print(f"Converted {input_path} to {output_path}")

def select_file():
    # Create a file dialog to select the input file
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a NIfTI file", filetypes=[("NIfTI files", "*.nii")])
    
    if file_path:
        convert_nii_to_niigz(file_path)
    else:
        print("No file selected")

if __name__ == "__main__":
    select_file()
