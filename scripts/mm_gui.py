import customtkinter as ctk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys
import threading
import urllib.request
import io

ctk.set_appearance_mode("light")

# MuscleMap brand colors
ORANGE        = "#f5a733"
PRIMARY       = "#f07c2c"
PRIMARY_HOVER = "#d96a1a"
PRIMARY_TEXT  = "#ffffff"
BG            = "#f0f2f5"
SIDEBAR_BG    = "#ffffff"
CARD          = "#ffffff"
TEXT          = "#1a1a2e"
MUTED         = "#6b6875"
BORDER        = "#ddd8e4"
FONT          = "Segoe UI"

FAVICON_URL  = "https://musclemap.github.io/MuscleMap/favicon.ico"
GIF_URL      = "https://musclemap.github.io/MuscleMap/assets/images/musclemap_scroll.gif"

# If user saves the colorful body image locally, use that instead of the GIF
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LOCAL_HERO   = os.path.join(_SCRIPT_DIR, "hero_body.png")

REGIONS = ['wholebody', 'abdomen', 'pelvis', 'thigh', 'leg']


# ---------------------------------------------------------------------------
# Asset helpers
# ---------------------------------------------------------------------------
def _load_logo(size=(44, 44)):
    try:
        from PIL import Image
        with urllib.request.urlopen(FAVICON_URL, timeout=5) as r:
            data = r.read()
        ico = Image.open(io.BytesIO(data)).convert("RGBA")
        px = ico.load()
        for y in range(ico.height):
            for x in range(ico.width):
                r_, g, b, _ = px[x, y]
                if r_ < 45 and g < 45 and b < 45:
                    px[x, y] = (0, 0, 0, 0)
        ico = ico.resize(size, Image.LANCZOS)
        return ctk.CTkImage(light_image=ico, dark_image=ico, size=size)
    except Exception:
        return None


def _load_static_hero(path, max_w=300, max_h=360):
    """Load a local PNG, scale proportionally to fit max_w x max_h, no distortion."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        return ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
    except Exception:
        return None


def _load_gif_frames(url, display_size, step=2, timeout=12):
    """
    Download animated GIF, strip black background, return list of
    (CTkImage, delay_ms) tuples sampled every `step` frames.
    """
    try:
        import numpy as np
        from PIL import Image
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        gif = Image.open(io.BytesIO(data))
        n = getattr(gif, "n_frames", 1)
        frames = []
        for i in range(0, n, step):
            gif.seek(i)
            frame = gif.copy().convert("RGBA")
            arr = np.array(frame)
            # Remove near-black canvas so the scan floats on the white card
            dark = (arr[:, :, 0] < 25) & (arr[:, :, 1] < 25) & (arr[:, :, 2] < 25)
            arr[dark, 3] = 0
            frame = Image.fromarray(arr)
            frame = frame.resize(display_size, Image.LANCZOS)
            delay = gif.info.get("duration", 80) * step
            cimg = ctk.CTkImage(light_image=frame, dark_image=frame, size=display_size)
            frames.append((cimg, max(delay, 40)))
        return frames
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class MuscleMapApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MuscleMap Toolbox")
        self.geometry("1060x680")
        self.minsize(860, 560)
        self.configure(fg_color=BG)

        self.logo_img    = None
        self.static_hero = None
        self._anim_frames: list = []
        self._anim_idx   = 0
        self._anim_job   = None
        self._anim_on    = False

        threading.Thread(target=self._preload_assets, daemon=True).start()
        self._build_hero()

    # ------------------------------------------------------------------
    # Asset pre-loading (background thread)
    # ------------------------------------------------------------------
    def _preload_assets(self):
        self.logo_img = _load_logo()

        if os.path.exists(LOCAL_HERO):
            self.static_hero = _load_static_hero(LOCAL_HERO)
        else:
            self._anim_frames = _load_gif_frames(GIF_URL, (270, 270))

        self.after(0, self._on_assets_ready)

    def _on_assets_ready(self):
        if hasattr(self, "_hero_logo_lbl") and self.logo_img:
            self._hero_logo_lbl.configure(image=self.logo_img)

        if not hasattr(self, "_hero_body_lbl"):
            return

        if self.static_hero:
            self._hero_body_lbl.configure(image=self.static_hero, text="")
        elif self._anim_frames:
            self._anim_on = True
            self._hero_body_lbl.configure(text="")
            self._tick()

    def _tick(self):
        if not self._anim_on or not self._anim_frames:
            return
        if not hasattr(self, "_hero_body_lbl"):
            return
        img, delay = self._anim_frames[self._anim_idx]
        self._hero_body_lbl.configure(image=img)
        self._anim_idx = (self._anim_idx + 1) % len(self._anim_frames)
        self._anim_job = self.after(delay, self._tick)

    def _stop_anim(self):
        self._anim_on = False
        if self._anim_job:
            self.after_cancel(self._anim_job)

    # ------------------------------------------------------------------
    # Hero page
    # ------------------------------------------------------------------
    def _build_hero(self):
        self._hero_frame = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self._hero_frame.pack(fill="both", expand=True)

        # Top bar
        bar = ctk.CTkFrame(self._hero_frame, fg_color=CARD, height=56,
                           corner_radius=0, border_width=1, border_color=BORDER)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        logo_row = ctk.CTkFrame(bar, fg_color="transparent")
        logo_row.pack(side="left", padx=22, pady=8)
        self._hero_logo_lbl = ctk.CTkLabel(logo_row, text="", image=self.logo_img)
        self._hero_logo_lbl.pack(side="left")
        ctk.CTkLabel(
            logo_row, text="MuscleMap",
            font=ctk.CTkFont(family=FONT, size=16, weight="bold"), text_color=TEXT,
        ).pack(side="left", padx=(10, 0))

        # Centred content area
        outer = ctk.CTkFrame(self._hero_frame, fg_color=BG, corner_radius=0)
        outer.pack(fill="both", expand=True)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        # Hero card
        card = ctk.CTkFrame(outer, fg_color=CARD, corner_radius=20,
                            border_width=1, border_color=BORDER)
        card.grid(row=0, column=0, padx=50, pady=36, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=0)
        card.grid_rowconfigure(0, weight=1)

        # --- Left: text ---
        left = ctk.CTkFrame(card, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(44, 16), pady=44)
        left.grid_rowconfigure(99, weight=1)   # spacer at bottom

        ctk.CTkLabel(
            left, text="OPEN-SOURCE TOOLBOX",
            font=ctk.CTkFont(family=FONT, size=11, weight="bold"),
            text_color=MUTED,
        ).pack(anchor="w")

        ctk.CTkLabel(
            left, text="MuscleMap",
            font=ctk.CTkFont(family=FONT, size=44, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", pady=(8, 0))

        ctk.CTkLabel(
            left,
            text="Whole-body muscle segmentation and\nquantitative analysis for large-scale\nimaging studies.",
            font=ctk.CTkFont(family=FONT, size=14),
            text_color=MUTED, justify="left",
        ).pack(anchor="w", pady=(10, 30))

        # Buttons
        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(anchor="w")
        ctk.CTkButton(
            btn_row, text="Get started", height=44, width=155,
            fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
            text_color=PRIMARY_TEXT,
            font=ctk.CTkFont(family=FONT, size=14, weight="bold"),
            corner_radius=22, command=self._launch_app,
        ).pack(side="left", padx=(0, 14))
        ctk.CTkButton(
            btn_row, text="View code", height=44, width=135,
            fg_color="transparent", hover_color="#eeecf4",
            text_color=TEXT, border_color=BORDER, border_width=2,
            font=ctk.CTkFont(family=FONT, size=14), corner_radius=22,
            command=lambda: os.startfile("https://github.com/MuscleMap/MuscleMap"),
        ).pack(side="left")

        ctk.CTkLabel(
            left,
            text="Built for researchers, clinicians, and data scientists\nworking with MRI and CT.",
            font=ctk.CTkFont(family=FONT, size=12), text_color=MUTED, justify="left",
        ).pack(anchor="w", pady=(20, 0))

        # --- Right: image / animation ---
        right = ctk.CTkFrame(card, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._hero_body_lbl = ctk.CTkLabel(
            right, text="Loading...",
            text_color=MUTED, font=ctk.CTkFont(family=FONT, size=13),
        )
        self._hero_body_lbl.grid(row=0, column=0, padx=(16, 36), pady=36)

    # ------------------------------------------------------------------
    # Hero → main app
    # ------------------------------------------------------------------
    def _launch_app(self):
        self._stop_anim()
        self._hero_frame.destroy()
        self._build_main()

    def _build_main(self):
        self.sidebar = ctk.CTkFrame(self, width=210, fg_color=SIDEBAR_BG,
                                    corner_radius=0, border_width=1, border_color=BORDER)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        right = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        right.pack(side="left", fill="both", expand=True)
        self._build_log(right)

        self.panels_frame = ctk.CTkFrame(right, fg_color=BG, corner_radius=0)
        self.panels_frame.pack(fill="both", expand=True, padx=24, pady=(24, 10))

        self.panels: dict[str, ctk.CTkScrollableFrame] = {
            "segment":  self._build_segment_panel(),
            "extract":  self._build_extract_panel(),
            "workflow": self._build_workflow_panel(),
        }
        self.show_panel("segment")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    def _build_sidebar(self):
        logo_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_row.pack(pady=(20, 2), padx=14, fill="x")
        if self.logo_img:
            ctk.CTkLabel(logo_row, image=self.logo_img, text="").pack(side="left")
        ctk.CTkLabel(
            logo_row, text="MuscleMap",
            font=ctk.CTkFont(family=FONT, size=14, weight="bold"), text_color=TEXT,
        ).pack(side="left", padx=(8, 0))

        ctk.CTkLabel(
            self.sidebar, text="Toolbox",
            font=ctk.CTkFont(family=FONT, size=11), text_color=MUTED,
        ).pack(anchor="w", padx=20, pady=(0, 14))

        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(fill="x", padx=14, pady=(0, 12))

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, label in [("segment", "Segmentation"), ("extract", "Extract Metrics"), ("workflow", "Workflow")]:
            btn = ctk.CTkButton(
                self.sidebar, text=label, height=40,
                fg_color="transparent", hover_color="#fdf0e8",
                text_color=MUTED, anchor="w", corner_radius=8,
                font=ctk.CTkFont(family=FONT, size=13),
                command=lambda k=key: self.show_panel(k),
            )
            btn.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[key] = btn

        ctk.CTkLabel(
            self.sidebar, text="v1.3",
            font=ctk.CTkFont(family=FONT, size=11), text_color=MUTED,
        ).pack(side="bottom", pady=18)

    def show_panel(self, key: str):
        for k, btn in self.nav_buttons.items():
            btn.configure(
                fg_color="#fff3e8" if k == key else "transparent",
                text_color=PRIMARY if k == key else MUTED,
            )
        for k, panel in self.panels.items():
            panel.pack(fill="both", expand=True) if k == key else panel.pack_forget()

    # ------------------------------------------------------------------
    # Output log
    # ------------------------------------------------------------------
    def _build_log(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=12, height=145,
                            border_width=1, border_color=BORDER)
        card.pack(side="bottom", fill="x", padx=24, pady=(0, 16))
        card.pack_propagate(False)

        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(hdr, text="Output",
                     font=ctk.CTkFont(family=FONT, size=12, weight="bold"),
                     text_color=MUTED).pack(side="left")
        ctk.CTkButton(
            hdr, text="Clear", width=54, height=22,
            fg_color="transparent", hover_color="#f0eeee", text_color=MUTED,
            font=ctk.CTkFont(family=FONT, size=11),
            command=lambda: self.log_text.delete("1.0", "end"),
        ).pack(side="right")

        self.log_text = ctk.CTkTextbox(
            card, fg_color=CARD, text_color="#44434d",
            font=ctk.CTkFont(family="Consolas", size=11), corner_radius=0,
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _log(self, text: str):
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
            font=ctk.CTkFont(family=FONT, size=20, weight="bold"), text_color=TEXT,
        ).pack(anchor="w", pady=(0, 20))

    def _field(self, parent, label: str, browse: str = "file") -> ctk.CTkEntry:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text=label, width=165, anchor="w",
                     text_color=MUTED, font=ctk.CTkFont(family=FONT, size=13)).pack(side="left")
        entry = ctk.CTkEntry(
            row, placeholder_text="...", fg_color=CARD,
            border_color=BORDER, border_width=1, height=34, text_color=TEXT,
            font=ctk.CTkFont(family=FONT, size=13),
        )
        entry.pack(side="left", fill="x", expand=True, padx=(8, 8))
        cmd = (lambda e=entry: self._browse_file(e)) if browse == "file" \
              else (lambda e=entry: self._browse_dir(e))
        ctk.CTkButton(row, text="Browse", width=84, height=34,
                      fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                      text_color=PRIMARY_TEXT,
                      font=ctk.CTkFont(family=FONT, size=13),
                      command=cmd).pack(side="left")
        return entry

    def _option(self, parent, label: str, options: list) -> ctk.StringVar:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text=label, width=165, anchor="w",
                     text_color=MUTED, font=ctk.CTkFont(family=FONT, size=13)).pack(side="left")
        var = ctk.StringVar(value=options[0])
        ctk.CTkOptionMenu(
            row, variable=var, values=options, height=34,
            fg_color=CARD, button_color=PRIMARY, button_hover_color=PRIMARY_HOVER,
            dropdown_fg_color=CARD, dropdown_hover_color="#fff3e8",
            text_color=TEXT, dropdown_text_color=TEXT,
            font=ctk.CTkFont(family=FONT, size=13),
        ).pack(side="left", padx=(8, 0))
        return var

    def _gpu_row(self, parent) -> ctk.StringVar:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text="Use GPU:", width=165, anchor="w",
                     text_color=MUTED, font=ctk.CTkFont(family=FONT, size=13)).pack(side="left")
        var = ctk.StringVar(value="Y")
        for txt, val, pad in [("Yes", "Y", (8, 16)), ("No", "N", (0, 0))]:
            ctk.CTkRadioButton(row, text=txt, variable=var, value=val,
                               fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                               text_color=TEXT, font=ctk.CTkFont(family=FONT, size=13),
                               ).pack(side="left", padx=pad)
        return var

    def _components_row(self, parent) -> ctk.StringVar:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text="Components:", width=165, anchor="w",
                     text_color=MUTED, font=ctk.CTkFont(family=FONT, size=13)).pack(side="left")
        var = ctk.StringVar(value="2")
        for txt, val, pad in [("2", "2", (8, 16)), ("3", "3", (0, 0))]:
            ctk.CTkRadioButton(row, text=txt, variable=var, value=val,
                               fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                               text_color=TEXT, font=ctk.CTkFont(family=FONT, size=13),
                               ).pack(side="left", padx=pad)
        return var

    def _run_btn(self, parent, cmd):
        ctk.CTkButton(
            parent, text="Run", height=44, width=160,
            fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
            text_color=PRIMARY_TEXT, corner_radius=22,
            font=ctk.CTkFont(family=FONT, size=15, weight="bold"),
            command=cmd,
        ).pack(pady=(28, 8))

    @staticmethod
    def _browse_file(entry):
        path = filedialog.askopenfilename()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    @staticmethod
    def _browse_dir(entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _script(self, name: str) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    def _run_commands(self, steps: list, success_msg: str):
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
    # Panels
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
        cmd = [sys.executable, self._script("mm_segment.py"),
               "-i", self._seg_input.get(), "-r", self._seg_region.get(),
               "-g", self._seg_gpu.get()]
        if self._seg_output.get():
            cmd += ["-o", self._seg_output.get()]
        self._run_commands([cmd], "Segmentation completed.")

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
        cmd = [sys.executable, self._script("mm_extract_metrics.py"),
               "-m", method, "-r", self._ext_region.get(), "-o", self._ext_output.get()]
        if self._ext_seg.get():
            cmd += ["-s", self._ext_seg.get()]
        if method == "dixon":
            if self._ext_fat.get():   cmd += ["-f", self._ext_fat.get()]
            if self._ext_water.get(): cmd += ["-w", self._ext_water.get()]
        else:
            if self._ext_input.get():      cmd += ["-i", self._ext_input.get()]
            if self._ext_components.get(): cmd += ["-c", self._ext_components.get()]
        self._run_commands([cmd], "Extraction completed.")

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
        method, region = self._wf_method.get(), self._wf_region.get()
        output_dir, use_gpu = self._wf_output.get(), self._wf_gpu.get()
        seg_s, ext_s = self._script("mm_segment.py"), self._script("mm_extract_metrics.py")

        if method == "dixon":
            fat, water = self._wf_fat.get(), self._wf_water.get()
            seg_out = os.path.join(output_dir,
                os.path.basename(water).replace(".nii.gz", "_dseg.nii.gz").replace(".nii", "_dseg.nii.gz"))
            steps = [
                [sys.executable, seg_s, "-i", water, "-r", region, "-g", use_gpu, "-o", output_dir],
                [sys.executable, seg_s, "-i", fat,   "-r", region, "-g", use_gpu, "-o", output_dir],
                [sys.executable, ext_s, "-m", "dixon", "-r", region, "-o", output_dir,
                 "-f", fat, "-w", water, "-s", seg_out],
            ]
        else:
            inp = self._wf_input.get()
            seg_out = os.path.join(output_dir,
                os.path.basename(inp).replace(".nii.gz", "_dseg.nii.gz").replace(".nii", "_dseg.nii.gz"))
            seg_cmd = [sys.executable, seg_s, "-i", inp, "-r", region, "-g", use_gpu, "-o", output_dir]
            ext_cmd = [sys.executable, ext_s, "-m", method, "-r", region, "-o", output_dir,
                       "-i", inp, "-s", seg_out]
            if method != "average":
                ext_cmd += ["-c", self._wf_components.get()]
            steps = [seg_cmd, ext_cmd]

        self._run_commands(steps, "Workflow completed.")


def main():
    app = MuscleMapApp()
    app.mainloop()


if __name__ == "__main__":
    main()
