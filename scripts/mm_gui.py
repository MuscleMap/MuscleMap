import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import subprocess
import os
import sys
import threading
import random
import math

ctk.set_appearance_mode("light")

# MuscleMap brand colors
ORANGE        = "#f5a733"
PRIMARY       = "#f07c2c"
PRIMARY_HOVER = "#d96a1a"
PRIMARY_TEXT  = "#ffffff"
BG            = "#fffaf6"
SIDEBAR_BG    = "#fff3e8"
CARD          = "#ffffff"
TEXT          = "#1a1a2e"
MUTED         = "#6b6875"
BORDER        = "#ddd8e4"
FONT          = "Segoe UI"

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_SCRIPT_DIR)
_ASSETS_DIR   = os.path.join(_ROOT_DIR, "assets")
LOCAL_LOGO    = os.path.join(_ASSETS_DIR, "logo_musclemap_white.png")
LOCAL_FAVICON = os.path.join(_ASSETS_DIR, "favicon.png")

REGIONS = ['wholebody', 'abdomen', 'pelvis', 'thigh', 'leg']

# ---------------------------------------------------------------------------
# Asset helpers
# ---------------------------------------------------------------------------
def _load_image(path, size):
    try:
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        img = img.resize(size, Image.LANCZOS)
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
    except Exception:
        return None


_ICO_PATH_GUI  = None
_MODELS_DIR_GUI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _cached_model_version(region: str) -> str:
    """Return version string of highest cached model for region, or None."""
    versions = _cached_versions_for_region(region)
    return versions[-1] if versions else None


def _cached_versions_for_region(region: str) -> list:
    """Return sorted list of cached version strings for region (e.g. ['1.3'])."""
    region_dir = os.path.join(_MODELS_DIR_GUI, region)
    if not os.path.isdir(region_dir):
        return []
    return sorted(
        d[1:] for d in os.listdir(region_dir)
        if os.path.isdir(os.path.join(region_dir, d)) and d.startswith("v")
    )

def _set_window_icon(window):
    global _ICO_PATH_GUI
    try:
        if _ICO_PATH_GUI is None:
            import tempfile
            from PIL import Image
            fav = Image.open(LOCAL_FAVICON)
            _ICO_PATH_GUI = os.path.join(tempfile.gettempdir(), "musclemap_icon.ico")
            fav.save(_ICO_PATH_GUI, format="ICO", sizes=[(32, 32), (16, 16)])
        window.iconbitmap(_ICO_PATH_GUI)
    except Exception:
        pass


def _load_hero_logo(max_w=320, max_h=380):
    try:
        from PIL import Image
        img = Image.open(LOCAL_LOGO).convert("RGBA")
        w, h = img.size
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        return ctk.CTkImage(light_image=img, dark_image=img, size=(nw, nh))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task window  (spinner → confetti success / error)
# ---------------------------------------------------------------------------
class _TaskWindow(ctk.CTkToplevel):
    W, H = 360, 260
    _CONFETTI_COLORS = ["#f07c2c", "#e05a4e", "#f5a733", "#5b9bd5", "#4caf6e", "#9c5bb5"]

    def __init__(self, parent, label="Processing…", subtitle=""):
        super().__init__(parent)
        self.title("MuscleMap")
        self.resizable(False, False)
        self.configure(fg_color=BG)
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.grab_set()
        self.after(200, lambda: _set_window_icon(self))
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{self.W}x{self.H}+{(sw - self.W)//2}+{(sh - self.H)//2}")

        self._running = True
        self._angle   = 0

        self._canvas = tk.Canvas(self, width=72, height=72, bg=BG, highlightthickness=0)
        self._canvas.pack(pady=(28, 8))

        self._lbl = ctk.CTkLabel(self, text=label,
            font=ctk.CTkFont(family=FONT, size=14, weight="bold"), text_color=TEXT)
        self._lbl.pack()

        if subtitle:
            ctk.CTkLabel(self, text=subtitle,
                font=ctk.CTkFont(family=FONT, size=12, weight="bold"),
                text_color=PRIMARY).pack(pady=(2, 0))

        ctk.CTkLabel(self, text="This may take a few minutes…",
            font=ctk.CTkFont(family=FONT, size=11), text_color=MUTED).pack(pady=(4, 0))

        self._spin()

    def _spin(self):
        if not self._running:
            return
        c = self._canvas
        c.delete("all")
        cx = cy = 36
        r = 26
        c.create_arc(cx-r, cy-r, cx+r, cy+r, start=0, extent=359,
                     outline="#fde8d0", width=5, style="arc")
        c.create_arc(cx-r, cy-r, cx+r, cy+r, start=self._angle, extent=270,
                     outline=PRIMARY, width=5, style="arc")
        self._angle = (self._angle + 9) % 360
        self.after(25, self._spin)

    def show_success(self, message="Completed successfully."):
        self._running = False
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        for w in self.winfo_children():
            w.destroy()

        conf = tk.Canvas(self, width=self.W, height=self.H, bg=BG, highlightthickness=0)
        conf.place(x=0, y=0)

        ctk.CTkLabel(self, text="✓",
            font=ctk.CTkFont(family=FONT, size=52, weight="bold"),
            text_color=PRIMARY, fg_color="transparent",
        ).place(relx=0.5, rely=0.27, anchor="center")

        ctk.CTkLabel(self, text=message,
            font=ctk.CTkFont(family=FONT, size=13, weight="bold"),
            text_color=TEXT, fg_color="transparent",
        ).place(relx=0.5, rely=0.54, anchor="center")

        ctk.CTkButton(self, text="OK", width=110, height=36,
            fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
            text_color=PRIMARY_TEXT, corner_radius=18,
            font=ctk.CTkFont(family=FONT, size=13, weight="bold"),
            command=self.destroy,
        ).place(relx=0.5, rely=0.80, anchor="center")

        particles = [
            {"x": random.randint(0, self.W), "y": random.randint(-60, -4),
             "vx": random.uniform(-1.2, 1.2), "vy": random.uniform(2.5, 5.5),
             "color": random.choice(self._CONFETTI_COLORS),
             "size": random.randint(6, 11),
             "shape": random.choice(["rect", "oval"])}
            for _ in range(70)
        ]

        def _animate():
            conf.delete("all")
            for p in particles:
                p["x"] += p["vx"]; p["y"] += p["vy"]
                p["vy"] = min(p["vy"] + 0.1, 8)
                if p["y"] > self.H:
                    p["y"] = -8; p["x"] = random.randint(0, self.W)
                    p["vy"] = random.uniform(2.5, 5.5)
                x, y, s = int(p["x"]), int(p["y"]), p["size"]
                if p["shape"] == "rect":
                    conf.create_rectangle(x, y, x+s, y+s//2, fill=p["color"], outline="")
                else:
                    conf.create_oval(x, y, x+s, y+s, fill=p["color"], outline="")
            if self.winfo_exists():
                conf.after(30, _animate)

        _animate()

    def show_error(self, message):
        self._running = False
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=BG)

        ctk.CTkLabel(self, text="✕",
            font=ctk.CTkFont(family=FONT, size=52, weight="bold"),
            text_color="#e05a4e", fg_color="transparent",
        ).place(relx=0.5, rely=0.27, anchor="center")

        ctk.CTkLabel(self, text="An error occurred.",
            font=ctk.CTkFont(family=FONT, size=13, weight="bold"),
            text_color=TEXT, fg_color="transparent",
        ).place(relx=0.5, rely=0.52, anchor="center")

        ctk.CTkLabel(self, text=message[:120],
            font=ctk.CTkFont(family=FONT, size=10), text_color=MUTED,
            wraplength=300, fg_color="transparent",
        ).place(relx=0.5, rely=0.65, anchor="center")

        ctk.CTkButton(self, text="Close", width=110, height=36,
            fg_color="#e05a4e", hover_color="#c94a3e",
            text_color=PRIMARY_TEXT, corner_radius=18,
            font=ctk.CTkFont(family=FONT, size=13, weight="bold"),
            command=self.destroy,
        ).place(relx=0.5, rely=0.83, anchor="center")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class MuscleMapApp(ctk.CTk):
    def __init__(self):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("musclemap.toolbox.1")
        except Exception:
            pass
        super().__init__()
        self.title("MuscleMap Toolbox")
        self.geometry("1060x680")
        self.minsize(860, 560)
        self.configure(fg_color=BG)
        try:
            import tempfile
            from PIL import Image
            _fav = Image.open(LOCAL_FAVICON)
            _ico_path = os.path.join(tempfile.gettempdir(), "musclemap_icon.ico")
            _fav.save(_ico_path, format="ICO", sizes=[(32, 32), (16, 16)])
            self.iconbitmap(_ico_path)
        except Exception:
            pass

        self.favicon_img = _load_image(LOCAL_FAVICON, (32, 32))
        self.hero_img    = _load_hero_logo()

        self._build_hero()

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
        ctk.CTkLabel(logo_row, text="", image=self.favicon_img).pack(side="left")
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
        card.grid(row=0, column=0, padx=50, pady=36, sticky="ns")

        # --- Right: logo (packed first so it stays right) ---
        right = ctk.CTkFrame(card, fg_color="transparent")
        right.pack(side="right", padx=(0, 36), pady=36)
        ctk.CTkLabel(right, text="", image=self.hero_img).pack()

        # --- Left: text ---
        left = ctk.CTkFrame(card, fg_color="transparent")
        left.pack(side="left", padx=(44, 24), pady=44)

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


    # ------------------------------------------------------------------
    # Hero → main app
    # ------------------------------------------------------------------
    def _launch_app(self):
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
    def _back_to_hero(self):
        for w in self.winfo_children():
            w.destroy()
        self._build_hero()

    def _build_sidebar(self):
        logo_row = ctk.CTkFrame(self.sidebar, fg_color="transparent", cursor="hand2")
        logo_row.pack(pady=(20, 2), padx=14, fill="x")
        logo_row.bind("<Button-1>", lambda _: self._back_to_hero())
        if self.favicon_img:
            lbl_icon = ctk.CTkLabel(logo_row, image=self.favicon_img, text="")
            lbl_icon.pack(side="left")
            lbl_icon.bind("<Button-1>", lambda _: self._back_to_hero())
        lbl_text = ctk.CTkLabel(
            logo_row, text="MuscleMap",
            font=ctk.CTkFont(family=FONT, size=14, weight="bold"), text_color=TEXT,
        )
        lbl_text.pack(side="left", padx=(8, 0))
        lbl_text.bind("<Button-1>", lambda _: self._back_to_hero())

        ctk.CTkLabel(
            self.sidebar, text="Toolbox",
            font=ctk.CTkFont(family=FONT, size=11), text_color=MUTED,
        ).pack(anchor="w", padx=20, pady=(0, 14))

        ctk.CTkFrame(self.sidebar, height=1, fg_color=BORDER).pack(fill="x", padx=14, pady=(0, 12))

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, label in [("segment", "Segmentation"), ("extract", "Extract Metrics"), ("workflow", "Full MuscleMap Workflow")]:
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
        return self._components_row_into(row)

    def _components_row_into(self, row) -> ctk.StringVar:
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

    def _run_commands(self, steps: list, success_msg: str, subtitle: str = ""):
        win = _TaskWindow(self, subtitle=subtitle)

        def _thread():
            for cmd in steps:
                self._log("$ " + " ".join(cmd))
                try:
                    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    if proc.stdout:
                        self._log(proc.stdout.strip())
                except subprocess.CalledProcessError as e:
                    err = e.stderr.strip() or str(e)
                    self._log(f"[ERROR] {err}")
                    self.after(0, lambda m=err: win.show_error(m))
                    return
            self._log(f"[OK] {success_msg}")
            self.after(0, lambda: win.show_success(success_msg))

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

        # Version dropdown — values update when region changes
        ver_row = ctk.CTkFrame(p, fg_color="transparent")
        ver_row.pack(fill="x", pady=5)
        ctk.CTkLabel(ver_row, text="Model version:", width=165, anchor="w",
                     text_color=MUTED, font=ctk.CTkFont(family=FONT, size=13)).pack(side="left")
        self._seg_version_var = ctk.StringVar(value="latest")
        self._seg_version_menu = ctk.CTkOptionMenu(
            ver_row, variable=self._seg_version_var, values=["latest"],
            height=34, fg_color=CARD, button_color=PRIMARY,
            button_hover_color=PRIMARY_HOVER, dropdown_fg_color=CARD,
            dropdown_hover_color="#fff3e8", text_color=TEXT,
            dropdown_text_color=TEXT, font=ctk.CTkFont(family=FONT, size=13),
        )
        self._seg_version_menu.pack(side="left", padx=(8, 0))

        def _update_versions(*_):
            cached   = _cached_versions_for_region(self._seg_region.get())
            versions = cached if cached else ["latest"]
            self._seg_version_menu.configure(values=versions)
            if self._seg_version_var.get() not in versions:
                self._seg_version_var.set(versions[-1] if cached else "latest")

        self._seg_region.trace_add("write", _update_versions)
        _update_versions()

        self._seg_gpu = self._gpu_row(p)
        self._run_btn(p, self._run_segmentation)
        return p

    def _run_segmentation(self):
        region  = self._seg_region.get()
        version = self._seg_version_var.get()
        cmd = [sys.executable, self._script("mm_segment.py"),
               "-i", self._seg_input.get(), "-r", region, "-g", self._seg_gpu.get()]
        if version != "latest":
            cmd += ["--model_version", version]
        if self._seg_output.get():
            cmd += ["-o", self._seg_output.get()]
        version_str = f"v{version}" if version != "latest" else (
            f"v{_cached_model_version(region)}" if _cached_model_version(region) else "Downloading…"
        )
        self._run_commands([cmd], "Segmentation completed.",
                           subtitle=f"{region.capitalize()}  |  {version_str}")

    def _build_extract_panel(self) -> ctk.CTkScrollableFrame:
        p = ctk.CTkScrollableFrame(self.panels_frame, fg_color=BG, corner_radius=0)
        self._heading(p, "Extract Metrics")
        self._ext_method = self._option(p, "Method:", ["dixon", "kmeans", "gmm", "average"])

        dyn = ctk.CTkFrame(p, fg_color="transparent")
        dyn.pack(fill="x")
        self._ext_input = self._field(dyn, "Input Image:")
        self._ext_fat   = self._field(dyn, "Fat Image (Dixon):")
        self._ext_water = self._field(dyn, "Water Image (Dixon):")

        self._ext_seg        = self._field(p, "Segmentation Image:")
        self._ext_components_row = ctk.CTkFrame(p, fg_color="transparent")
        self._ext_components = self._components_row_into(self._ext_components_row)

        self._ext_qc_row = ctk.CTkFrame(p, fg_color="transparent")
        self._ext_qc_var = ctk.BooleanVar(value=False)
        ctk.CTkLabel(self._ext_qc_row, text="QC mode:", width=165, anchor="w",
                     text_color=MUTED, font=ctk.CTkFont(family=FONT, size=13)).pack(side="left")
        ctk.CTkCheckBox(self._ext_qc_row, text="Adjust thresholds interactively",
                        variable=self._ext_qc_var,
                        fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                        text_color=TEXT, font=ctk.CTkFont(family=FONT, size=13),
                        checkmark_color=PRIMARY_TEXT).pack(side="left", padx=(8, 0))

        self._ext_region     = self._option(p, "Region:", REGIONS)
        self._ext_output     = self._field(p, "Output Directory:", browse="dir")
        self._ext_output.insert(0, os.getcwd())
        self._run_btn(p, self._run_extraction)

        def _update_ext_fields(*_):
            method = self._ext_method.get()
            is_dixon = method == "dixon"
            self._ext_input.master.pack_forget()
            self._ext_fat.master.pack_forget()
            self._ext_water.master.pack_forget()
            self._ext_components_row.pack_forget()
            self._ext_qc_row.pack_forget()
            if is_dixon:
                self._ext_fat.master.pack(fill="x", pady=5)
                self._ext_water.master.pack(fill="x", pady=5)
            else:
                self._ext_input.master.pack(fill="x", pady=5)
            if method in ("kmeans", "gmm"):
                self._ext_components_row.pack(fill="x", pady=5, after=self._ext_seg.master)
                self._ext_qc_row.pack(fill="x", pady=5, after=self._ext_components_row)

        self._ext_method.trace_add("write", _update_ext_fields)
        _update_ext_fields()
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
            if self._ext_qc_var.get():     cmd += ["--qc"]
        self._run_commands([cmd], "Extraction completed.")

    def _build_workflow_panel(self) -> ctk.CTkScrollableFrame:
        p = ctk.CTkScrollableFrame(self.panels_frame, fg_color=BG, corner_radius=0)
        self._heading(p, "Full MuscleMap Workflow")
        self._wf_method = self._option(p, "Quantification Method:", ["kmeans", "gmm", "dixon", "average"])

        dyn = ctk.CTkFrame(p, fg_color="transparent")
        dyn.pack(fill="x")
        self._wf_input = self._field(dyn, "Input Image:")
        self._wf_fat   = self._field(dyn, "Fat Image (Dixon):")
        self._wf_water = self._field(dyn, "Water Image (Dixon):")

        self._wf_region     = self._option(p, "Region:", REGIONS)
        self._wf_components = self._components_row(p)
        self._wf_output     = self._field(p, "Output Directory:", browse="dir")
        self._wf_output.insert(0, os.getcwd())
        self._wf_gpu        = self._gpu_row(p)
        self._run_btn(p, self._run_workflow)

        def _update_wf_fields(*_):
            is_dixon = self._wf_method.get() == "dixon"
            self._wf_input.master.pack_forget()
            self._wf_fat.master.pack_forget()
            self._wf_water.master.pack_forget()
            if is_dixon:
                self._wf_fat.master.pack(fill="x", pady=5)
                self._wf_water.master.pack(fill="x", pady=5)
            else:
                self._wf_input.master.pack(fill="x", pady=5)

        self._wf_method.trace_add("write", _update_wf_fields)
        _update_wf_fields()
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
