import customtkinter as ctk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys
import threading
import urllib.request
import io

ctk.set_appearance_mode("light")

PRIMARY       = "#7253ed"
PRIMARY_HOVER = "#5739ce"
PRIMARY_TEXT  = "#ffffff"
BG            = "#f5f6fa"
SIDEBAR       = "#ecebed"
CARD          = "#ffffff"
TEXT          = "#27262b"
MUTED         = "#5c5962"

FAVICON_URL = "https://musclemap.github.io/MuscleMap/favicon.ico"
REGIONS     = ['wholebody', 'abdomen', 'pelvis', 'thigh', 'leg']


class MuscleMapApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MuscleMap Toolbox")
        self.geometry("960x680")
        self.minsize(800, 560)
        self.configure(fg_color=BG)

        self.logo_image = self._load_logo()
        self._build_ui()
        self.show_panel("segment")

    # ------------------------------------------------------------------
    # Logo
    # ------------------------------------------------------------------
    def _load_logo(self):
        try:
            from PIL import Image
            with urllib.request.urlopen(FAVICON_URL, timeout=4) as resp:
                data = resp.read()
            ico = Image.open(io.BytesIO(data))
            # Pick the largest frame available in the ICO
            best = ico.copy().convert("RGBA")
            for i in range(1, getattr(ico, "n_frames", 1)):
                try:
                    ico.seek(i)
                    frame = ico.convert("RGBA")
                    if frame.size[0] > best.size[0]:
                        best = frame
                except EOFError:
                    break
            best = best.resize((48, 48), Image.LANCZOS)
            return ctk.CTkImage(light_image=best, dark_image=best, size=(48, 48))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Layout skeleton
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.sidebar = ctk.CTkFrame(self, width=200, fg_color=SIDEBAR, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        right = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        right.pack(side="left", fill="both", expand=True)

        self._build_log(right)

        self.panels_frame = ctk.CTkFrame(right, fg_color=BG, corner_radius=0)
        self.panels_frame.pack(fill="both", expand=True, padx=20, pady=(20, 10))

        self.panels: dict[str, ctk.CTkScrollableFrame] = {
            "segment":  self._build_segment_panel(),
            "extract":  self._build_extract_panel(),
            "workflow": self._build_workflow_panel(),
        }

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    def _build_sidebar(self):
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(pady=(20, 0), padx=12, fill="x")

        if self.logo_image:
            ctk.CTkLabel(logo_frame, image=self.logo_image, text="").pack(side="left")
            ctk.CTkLabel(
                logo_frame, text="MuscleMap",
                font=ctk.CTkFont(size=15, weight="bold"), text_color=TEXT,
            ).pack(side="left", padx=(8, 0))
        else:
            ctk.CTkLabel(
                logo_frame, text="MuscleMap",
                font=ctk.CTkFont(size=16, weight="bold"), text_color=PRIMARY,
            ).pack(side="left")

        ctk.CTkLabel(
            self.sidebar, text="Toolbox",
            font=ctk.CTkFont(size=11), text_color=MUTED,
        ).pack(anchor="w", padx=18, pady=(0, 24))

        # Thin divider
        ctk.CTkFrame(self.sidebar, height=1, fg_color="#d8d5db").pack(fill="x", padx=12, pady=(0, 12))

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        nav = [("segment", "Segmentation"), ("extract", "Extract Metrics"), ("workflow", "Workflow")]
        for key, label in nav:
            btn = ctk.CTkButton(
                self.sidebar, text=label, height=40,
                fg_color="transparent", hover_color=PRIMARY,
                text_color=TEXT,
                anchor="w", corner_radius=8,
                command=lambda k=key: self.show_panel(k),
            )
            btn.pack(fill="x", padx=12, pady=3)
            self.nav_buttons[key] = btn

        ctk.CTkLabel(
            self.sidebar, text="v1.3",
            font=ctk.CTkFont(size=11), text_color=MUTED,
        ).pack(side="bottom", pady=18)

    def show_panel(self, key: str):
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=PRIMARY, text_color=PRIMARY_TEXT)
            else:
                btn.configure(fg_color="transparent", text_color=TEXT)
        for k, panel in self.panels.items():
            panel.pack(fill="both", expand=True) if k == key else panel.pack_forget()

    # ------------------------------------------------------------------
    # Output log
    # ------------------------------------------------------------------
    def _build_log(self, parent):
        log_card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, height=150,
                                border_width=1, border_color="#e6e1e8")
        log_card.pack(side="bottom", fill="x", padx=20, pady=(0, 16))
        log_card.pack_propagate(False)

        header = ctk.CTkFrame(log_card, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(header, text="Output", font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=MUTED).pack(side="left")
        ctk.CTkButton(
            header, text="Clear", width=54, height=22,
            fg_color="transparent", hover_color="#e6e1e8", text_color=MUTED,
            font=ctk.CTkFont(size=11),
            command=lambda: self.log_text.delete("1.0", "end"),
        ).pack(side="right")

        self.log_text = ctk.CTkTextbox(
            log_card, fg_color=CARD, text_color="#44434d",
            font=ctk.CTkFont(family="Courier", size=11), corner_radius=0,
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _log(self, text: str, color=None):
        self.after(0, lambda: self._log_safe(text))

    def _log_safe(self, text: str):
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    # ------------------------------------------------------------------
    # Shared widget helpers
    # ------------------------------------------------------------------
    def _heading(self, parent, text: str):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT,
        ).pack(anchor="w", pady=(0, 18))

    def _field(self, parent, label: str, browse: str = "file", placeholder: str = "") -> ctk.CTkEntry:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text=label, width=150, anchor="w", text_color=MUTED,
                     font=ctk.CTkFont(size=13)).pack(side="left")
        entry = ctk.CTkEntry(
            row, placeholder_text=placeholder or "...",
            fg_color=CARD, border_color=PRIMARY, border_width=1, height=34,
            text_color=TEXT,
        )
        entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        cmd = (lambda e=entry: self._browse_file(e)) if browse == "file" \
              else (lambda e=entry: self._browse_dir(e))
        ctk.CTkButton(row, text="Browse", width=80, height=34,
                      fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                      text_color=PRIMARY_TEXT, command=cmd).pack(side="left")
        return entry

    def _option(self, parent, label: str, options: list[str]) -> ctk.StringVar:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text=label, width=150, anchor="w", text_color=MUTED,
                     font=ctk.CTkFont(size=13)).pack(side="left")
        var = ctk.StringVar(value=options[0])
        ctk.CTkOptionMenu(
            row, variable=var, values=options, height=34,
            fg_color=CARD, button_color=PRIMARY, button_hover_color=PRIMARY_HOVER,
            dropdown_fg_color=CARD, dropdown_hover_color=PRIMARY,
            text_color=TEXT, dropdown_text_color=TEXT,
        ).pack(side="left", padx=(8, 0))
        return var

    def _gpu_row(self, parent) -> ctk.StringVar:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text="Use GPU:", width=150, anchor="w", text_color=MUTED,
                     font=ctk.CTkFont(size=13)).pack(side="left")
        var = ctk.StringVar(value="Y")
        ctk.CTkRadioButton(row, text="Yes", variable=var, value="Y",
                           fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                           text_color=TEXT).pack(side="left", padx=(8, 16))
        ctk.CTkRadioButton(row, text="No", variable=var, value="N",
                           fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                           text_color=TEXT).pack(side="left")
        return var

    def _components_row(self, parent) -> ctk.StringVar:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text="Components:", width=150, anchor="w", text_color=MUTED,
                     font=ctk.CTkFont(size=13)).pack(side="left")
        var = ctk.StringVar(value="2")
        ctk.CTkRadioButton(row, text="2", variable=var, value="2",
                           fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                           text_color=TEXT).pack(side="left", padx=(8, 16))
        ctk.CTkRadioButton(row, text="3", variable=var, value="3",
                           fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                           text_color=TEXT).pack(side="left")
        return var

    def _run_btn(self, parent, cmd):
        ctk.CTkButton(
            parent, text="Run", height=44, width=160,
            fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
            text_color=PRIMARY_TEXT,
            font=ctk.CTkFont(size=15, weight="bold"), command=cmd,
        ).pack(pady=(24, 8))

    @staticmethod
    def _browse_file(entry: ctk.CTkEntry):
        path = filedialog.askopenfilename()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    @staticmethod
    def _browse_dir(entry: ctk.CTkEntry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _script(self, name: str) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    def _run_commands(self, steps: list[list[str]], success_msg: str):
        def _thread():
            for cmd in steps:
                self._log("$ " + " ".join(cmd))
                try:
                    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    if proc.stdout:
                        self._log(proc.stdout.strip())
                except subprocess.CalledProcessError as e:
                    self._log(f"[ERROR] {e.stderr.strip() or str(e)}")
                    self.after(0, lambda: messagebox.showerror("Error", str(e)))
                    return
            self._log(f"[OK] {success_msg}")
            self.after(0, lambda: messagebox.showinfo("Success", success_msg))
        threading.Thread(target=_thread, daemon=True).start()

    # ------------------------------------------------------------------
    # Segmentation panel
    # ------------------------------------------------------------------
    def _build_segment_panel(self) -> ctk.CTkScrollableFrame:
        p = ctk.CTkScrollableFrame(self.panels_frame, fg_color=BG, corner_radius=0)
        self._heading(p, "Segmentation")
        self._seg_input  = self._field(p, "Input Image:")
        self._seg_region = self._option(p, "Region:", REGIONS)
        self._seg_output = self._field(p, "Output Directory:", browse="dir")
        self._seg_output.insert(0, os.getcwd())
        self._seg_gpu    = self._gpu_row(p)
        self._run_btn(p, self._run_segmentation)
        return p

    def _run_segmentation(self):
        cmd = [
            sys.executable, self._script("mm_segment.py"),
            "-i", self._seg_input.get(),
            "-r", self._seg_region.get(),
            "-g", self._seg_gpu.get(),
        ]
        if self._seg_output.get():
            cmd += ["-o", self._seg_output.get()]
        self._run_commands([cmd], "Segmentation completed.")

    # ------------------------------------------------------------------
    # Extract Metrics panel
    # ------------------------------------------------------------------
    def _build_extract_panel(self) -> ctk.CTkScrollableFrame:
        p = ctk.CTkScrollableFrame(self.panels_frame, fg_color=BG, corner_radius=0)
        self._heading(p, "Extract Metrics")
        self._ext_method     = self._option(p, "Method:", ["dixon", "kmeans", "gmm", "average"])
        self._ext_input      = self._field(p, "Input Image:")
        self._ext_fat        = self._field(p, "Fat Image (Dixon):")
        self._ext_water      = self._field(p, "Water Image (Dixon):")
        self._ext_seg        = self._field(p, "Segmentation:")
        self._ext_components = self._components_row(p)
        self._ext_region     = self._option(p, "Region:", REGIONS)
        self._ext_output     = self._field(p, "Output Directory:", browse="dir")
        self._ext_output.insert(0, os.getcwd())
        self._run_btn(p, self._run_extraction)
        return p

    def _run_extraction(self):
        method = self._ext_method.get()
        cmd = [
            sys.executable, self._script("mm_extract_metrics.py"),
            "-m", method,
            "-r", self._ext_region.get(),
            "-o", self._ext_output.get(),
        ]
        if self._ext_seg.get():
            cmd += ["-s", self._ext_seg.get()]
        if method == "dixon":
            if self._ext_fat.get():   cmd += ["-f", self._ext_fat.get()]
            if self._ext_water.get(): cmd += ["-w", self._ext_water.get()]
        else:
            if self._ext_input.get():      cmd += ["-i", self._ext_input.get()]
            if self._ext_components.get(): cmd += ["-c", self._ext_components.get()]
        self._run_commands([cmd], "Extraction completed.")

    # ------------------------------------------------------------------
    # Workflow panel (segment + extract, all methods)
    # ------------------------------------------------------------------
    def _build_workflow_panel(self) -> ctk.CTkScrollableFrame:
        p = ctk.CTkScrollableFrame(self.panels_frame, fg_color=BG, corner_radius=0)
        self._heading(p, "Workflow  —  Segment + Extract")
        self._wf_method     = self._option(p, "Method:", ["kmeans", "gmm", "dixon", "average"])
        self._wf_input      = self._field(p, "Input Image:")
        self._wf_fat        = self._field(p, "Fat Image (Dixon):")
        self._wf_water      = self._field(p, "Water Image (Dixon):")
        self._wf_region     = self._option(p, "Region:", REGIONS)
        self._wf_components = self._components_row(p)
        self._wf_output     = self._field(p, "Output Directory:", browse="dir")
        self._wf_output.insert(0, os.getcwd())
        self._wf_gpu        = self._gpu_row(p)
        self._run_btn(p, self._run_workflow)
        return p

    def _run_workflow(self):
        method     = self._wf_method.get()
        region     = self._wf_region.get()
        output_dir = self._wf_output.get()
        use_gpu    = self._wf_gpu.get()
        seg_script = self._script("mm_segment.py")
        ext_script = self._script("mm_extract_metrics.py")

        if method == "dixon":
            fat   = self._wf_fat.get()
            water = self._wf_water.get()
            base  = os.path.basename(water)
            seg_out = os.path.join(
                output_dir,
                base.replace(".nii.gz", "_dseg.nii.gz").replace(".nii", "_dseg.nii.gz"),
            )
            steps = [
                [sys.executable, seg_script, "-i", water, "-r", region, "-g", use_gpu, "-o", output_dir],
                [sys.executable, seg_script, "-i", fat,   "-r", region, "-g", use_gpu, "-o", output_dir],
                [sys.executable, ext_script, "-m", "dixon", "-r", region, "-o", output_dir,
                 "-f", fat, "-w", water, "-s", seg_out],
            ]
        else:
            input_img = self._wf_input.get()
            base = os.path.basename(input_img)
            seg_out = os.path.join(
                output_dir,
                base.replace(".nii.gz", "_dseg.nii.gz").replace(".nii", "_dseg.nii.gz"),
            )
            seg_cmd = [sys.executable, seg_script, "-i", input_img,
                       "-r", region, "-g", use_gpu, "-o", output_dir]
            ext_cmd = [sys.executable, ext_script, "-m", method,
                       "-r", region, "-o", output_dir, "-i", input_img, "-s", seg_out]
            if method != "average":
                ext_cmd += ["-c", self._wf_components.get()]
            steps = [seg_cmd, ext_cmd]

        self._run_commands(steps, "Workflow completed.")


def main():
    app = MuscleMapApp()
    app.mainloop()


if __name__ == "__main__":
    main()
