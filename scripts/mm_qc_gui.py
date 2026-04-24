import os
import numpy as np
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("light")

PRIMARY       = "#f07c2c"
PRIMARY_HOVER = "#d96a1a"
PRIMARY_TEXT  = "#ffffff"
LOGO_RED      = "#e05a4e"
LOGO_RED_HOVER= "#c94a3e"
BG            = "#f0f2f5"
CARD          = "#ffffff"
TEXT          = "#1a1a2e"
MUTED         = "#6b6875"
BORDER        = "#ddd8e4"
FONT          = "Segoe UI"

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_SCRIPT_DIR)
_ASSETS_DIR   = os.path.join(_ROOT_DIR, "assets")
LOCAL_FAVICON = os.path.join(_ASSETS_DIR, "favicon.png")


def _load_favicon(size=(24, 24)):
    try:
        from PIL import Image
        img = Image.open(LOCAL_FAVICON).convert("RGBA").resize(size, Image.LANCZOS)
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
    except Exception:
        return None


_ICO_PATH = None

def _ensure_ico():
    global _ICO_PATH
    if _ICO_PATH is not None:
        return _ICO_PATH
    try:
        import tempfile
        from PIL import Image
        fav = Image.open(LOCAL_FAVICON)
        _ICO_PATH = os.path.join(tempfile.gettempdir(), "musclemap_icon.ico")
        fav.save(_ICO_PATH, format="ICO", sizes=[(32, 32), (16, 16)])
    except Exception:
        _ICO_PATH = None
    return _ICO_PATH


def _set_taskbar_icon(window):
    ico = _ensure_ico()
    if ico:
        try:
            window.iconbitmap(ico)
        except Exception:
            pass


class QCWindow(ctk.CTkToplevel):
    """
    thresholds : {label_int: (muscle_max, fat_min_or_None)}

    Public results (set on Accept):
        result_muscle_delta  float
        result_fat_delta     float
        result_erased_mask   np.ndarray bool, shape == img_array.shape
    """

    def __init__(self, parent, img_array, label_img, thresholds, components, anatomy_name=""):
        super().__init__(parent)
        self._anatomy_name = anatomy_name
        self.title(anatomy_name if anatomy_name else "MuscleMap QC")
        self.configure(fg_color=BG)
        self.resizable(True, True)
        self.grab_set()
        self.after(200, lambda: _set_taskbar_icon(self))

        self._img_array  = img_array
        self._label_img  = label_img
        self._thresholds = thresholds
        self._components = components

        group_voxels      = img_array[np.isin(label_img, list(thresholds.keys()))]
        self._vmin        = float(group_voxels.min())
        self._vmax        = float(group_voxels.max())
        self._half_range  = (self._vmax - self._vmin) / 2.0

        group_mask = np.isin(label_img, list(thresholds.keys()))
        present    = np.where(np.any(group_mask, axis=(0, 1)))[0]
        self._valid_slices = present if len(present) > 0 else np.array([img_array.shape[2] // 2])

        # Eraser state
        self._erased_mask   = np.zeros(img_array.shape, dtype=bool)
        self._erase_history = []          # list of np.ndarray snapshots
        self._erase_active  = False
        self._is_dragging   = False
        self._cursor_pos    = None        # (xdata, ydata) of last known mouse position
        self._cursor_patch  = None        # live matplotlib Circle patch

        # Public results
        self.result_muscle_delta = 0.0
        self.result_fat_delta    = 0.0
        self.result_erased_mask  = self._erased_mask
        self.result_quit         = False

        self._build()

        self._slice_var.set(len(self._valid_slices) // 2)
        self._muscle_delta_var.set(0.0)
        if components == 3:
            self._fat_delta_var.set(0.0)

        self._canvas.mpl_connect("button_press_event",   self._on_mouse_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_mouse_move)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._canvas.mpl_connect("scroll_event",         self._on_scroll)
        self._canvas.mpl_connect("axes_leave_event",     self._on_axes_leave)

        self._update_plot()

        self.update_idletasks()
        w, h = 1020, 660
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self):
        self._build_topbar()

        content = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        content.pack(fill="both", expand=True, padx=16, pady=12)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        fig_frame = ctk.CTkFrame(content, fg_color=CARD, corner_radius=10,
                                  border_width=1, border_color=BORDER)
        fig_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._fig    = Figure(facecolor=CARD, tight_layout=True)
        self._ax     = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        ctrl = ctk.CTkScrollableFrame(content, fg_color=CARD, corner_radius=10,
                                      scrollbar_button_color=BORDER,
                                      scrollbar_button_hover_color=MUTED)
        ctrl.grid(row=0, column=1, sticky="nsew")
        self._build_controls(ctrl)

    def _build_topbar(self):
        bar = ctk.CTkFrame(self, fg_color=CARD, height=56, corner_radius=0,
                           border_width=1, border_color=BORDER)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.pack(side="left", padx=16, pady=8)

        self._logo_img = _load_favicon(size=(36, 36))
        if self._logo_img:
            ctk.CTkLabel(left, text="", image=self._logo_img).pack(side="left")

        # "Quality Control" with two-tone gradient feel: "Quality" orange, "Control" logo-red
        qc_frame = ctk.CTkFrame(left, fg_color="transparent")
        qc_frame.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(
            qc_frame, text="Quality ",
            font=ctk.CTkFont(family=FONT, size=16, weight="bold"),
            text_color=PRIMARY,
        ).pack(side="left")
        ctk.CTkLabel(
            qc_frame, text="Control",
            font=ctk.CTkFont(family=FONT, size=16, weight="bold"),
            text_color=LOGO_RED,
        ).pack(side="left")

        right_text = self._anatomy_name if self._anatomy_name else f"Label {next(iter(self._thresholds))}"
        badge = ctk.CTkFrame(bar, fg_color=LOGO_RED, corner_radius=12)
        badge.pack(side="right", padx=16, pady=12)
        ctk.CTkLabel(
            badge, text=right_text,
            font=ctk.CTkFont(family=FONT, size=12, weight="bold"),
            text_color=PRIMARY_TEXT,
        ).pack(padx=12, pady=4)

    def _build_controls(self, parent):
        def _section_label(text):
            ctk.CTkLabel(parent, text=text,
                         font=ctk.CTkFont(family=FONT, size=13, weight="bold"),
                         text_color=TEXT, anchor="w").pack(fill="x", padx=16, pady=(14, 2))

        def _value_label():
            lbl = ctk.CTkLabel(parent, text="",
                               font=ctk.CTkFont(family=FONT, size=11), text_color=MUTED)
            lbl.pack(fill="x", padx=16, pady=(0, 4))
            return lbl

        def _divider():
            ctk.CTkFrame(parent, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=6)

        steps = max(int(self._half_range * 10), 200)

        # ── Slice ──────────────────────────────────────────────────────
        _section_label("Slice")
        self._slice_var = ctk.IntVar(value=0)
        ctk.CTkSlider(
            parent, from_=0, to=max(len(self._valid_slices) - 1, 1),
            number_of_steps=max(len(self._valid_slices) - 1, 1),
            variable=self._slice_var,
            fg_color=BORDER, progress_color=PRIMARY,
            button_color=PRIMARY, button_hover_color=PRIMARY_HOVER,
            command=lambda _: self._update_plot(),
        ).pack(fill="x", padx=16)
        self._slice_label = _value_label()

        _divider()

        # ── Fat threshold Δ ────────────────────────────────────────────
        _section_label("Fat threshold  Δ")
        self._muscle_delta_var = ctk.DoubleVar(value=0.0)
        ctk.CTkSlider(
            parent, from_=self._half_range, to=-self._half_range,
            number_of_steps=steps,
            variable=self._muscle_delta_var,
            fg_color=BORDER, progress_color=PRIMARY,
            button_color=PRIMARY, button_hover_color=PRIMARY_HOVER,
            command=lambda _: self._update_plot(),
        ).pack(fill="x", padx=16)
        self._muscle_delta_label = _value_label()

        if self._components == 3:
            _divider()
            _section_label("Fat min  Δ")
            self._fat_delta_var = ctk.DoubleVar(value=0.0)
            ctk.CTkSlider(
                parent, from_=self._half_range, to=-self._half_range,
                number_of_steps=steps,
                variable=self._fat_delta_var,
                fg_color=BORDER, progress_color=PRIMARY,
                button_color=PRIMARY, button_hover_color=PRIMARY_HOVER,
                command=lambda _: self._update_plot(),
            ).pack(fill="x", padx=16)
            self._fat_delta_label = _value_label()

        _divider()

        # ── Eraser ─────────────────────────────────────────────────────
        _section_label("Eraser")

        self._erase_btn = ctk.CTkButton(
            parent, text="Eraser  OFF", height=34,
            fg_color="transparent", hover_color="#eeecf4",
            text_color=TEXT, border_color=BORDER, border_width=2,
            corner_radius=17,
            font=ctk.CTkFont(family=FONT, size=13),
            command=self._toggle_erase,
        )
        self._erase_btn.pack(fill="x", padx=16, pady=(0, 6))

        ctk.CTkLabel(parent, text="Pen size",
                     font=ctk.CTkFont(family=FONT, size=11), text_color=MUTED,
                     anchor="w").pack(fill="x", padx=16)
        self._pen_size_var = ctk.IntVar(value=3)
        ctk.CTkSlider(
            parent, from_=1, to=20, number_of_steps=19,
            variable=self._pen_size_var,
            fg_color=BORDER, progress_color=PRIMARY,
            button_color=PRIMARY, button_hover_color=PRIMARY_HOVER,
        ).pack(fill="x", padx=16)
        self._pen_label = ctk.CTkLabel(parent, text="3 px",
                                        font=ctk.CTkFont(family=FONT, size=11), text_color=MUTED)
        self._pen_label.pack(fill="x", padx=16, pady=(0, 6))
        def _pen_changed(*_):
            self._pen_label.configure(text=f"{self._pen_size_var.get()} px")
            self._redraw_cursor()
        self._pen_size_var.trace_add("write", _pen_changed)

        ctk.CTkButton(
            parent, text="Undo", height=34,
            fg_color="transparent", hover_color="#eeecf4",
            text_color=TEXT, border_color=BORDER, border_width=2,
            corner_radius=17,
            font=ctk.CTkFont(family=FONT, size=13),
            command=self._undo,
        ).pack(fill="x", padx=16, pady=(0, 4))

        _divider()

        # ── Buttons ────────────────────────────────────────────────────
        ctk.CTkButton(
            parent, text="Reset thresholds", height=34,
            fg_color="transparent", hover_color="#eeecf4",
            text_color=TEXT, border_color=BORDER, border_width=2,
            corner_radius=17,
            font=ctk.CTkFont(family=FONT, size=13),
            command=self._reset,
        ).pack(fill="x", padx=16, pady=(4, 4))

        ctk.CTkButton(
            parent, text="Quit QC", height=34,
            fg_color="transparent", hover_color="#fdecea",
            text_color=LOGO_RED, border_color=LOGO_RED, border_width=2,
            corner_radius=17,
            font=ctk.CTkFont(family=FONT, size=13),
            command=self._confirm_quit,
        ).pack(fill="x", padx=16, pady=(0, 6))

        ctk.CTkButton(
            parent, text="Accept", height=44,
            fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
            text_color=PRIMARY_TEXT, corner_radius=22,
            font=ctk.CTkFont(family=FONT, size=14, weight="bold"),
            command=self._accept,
        ).pack(fill="x", padx=16, pady=(0, 16))

    # ------------------------------------------------------------------
    # Mouse / eraser
    # ------------------------------------------------------------------
    def _toggle_erase(self):
        self._erase_active = not self._erase_active
        if self._erase_active:
            self._erase_btn.configure(
                text="Eraser  ON",
                fg_color=PRIMARY, hover_color=PRIMARY_HOVER,
                text_color=PRIMARY_TEXT, border_width=0,
            )
        else:
            self._erase_btn.configure(
                text="Eraser  OFF",
                fg_color="transparent", hover_color="#eeecf4",
                text_color=TEXT, border_color=BORDER, border_width=2,
            )

    def _on_scroll(self, event):
        current = self._slice_var.get()
        max_idx = len(self._valid_slices) - 1
        if event.button == "up":
            new = min(current + 1, max_idx)
        else:
            new = max(current - 1, 0)
        if new != current:
            self._slice_var.set(new)
            self._update_plot()

    def _on_axes_leave(self, *_):
        self._cursor_pos = None
        self._redraw_cursor()

    def _on_mouse_press(self, event):
        if not self._erase_active or event.inaxes != self._ax or event.button != 1:
            return
        self._erase_history.append(self._erased_mask.copy())
        self._is_dragging = True
        self._apply_erase(event)

    def _on_mouse_move(self, event):
        if self._erase_active and event.inaxes == self._ax and event.xdata is not None:
            self._cursor_pos = (event.xdata, event.ydata)
        else:
            self._cursor_pos = None

        if self._erase_active and self._is_dragging and event.inaxes == self._ax:
            self._apply_erase(event)
        else:
            self._redraw_cursor()

    def _on_mouse_release(self, *_):
        self._is_dragging = False

    def _redraw_cursor(self):
        import matplotlib.patches as mpatches
        if self._cursor_patch is not None:
            try:
                self._cursor_patch.remove()
            except Exception:
                pass
            self._cursor_patch = None
        if self._cursor_pos is not None and self._erase_active:
            r = self._pen_size_var.get()
            patch = mpatches.Circle(
                self._cursor_pos, radius=r,
                fill=False, edgecolor=LOGO_RED, linewidth=2.0, linestyle="--",
            )
            self._ax.add_patch(patch)
            self._cursor_patch = patch
        self._canvas.draw_idle()

    def _apply_erase(self, event):
        if event.xdata is None or event.ydata is None:
            return
        vox_x = int(round(event.xdata))
        vox_y = int(round(event.ydata))
        z     = int(self._valid_slices[self._slice_var.get()])
        r     = self._pen_size_var.get()

        sx, sy = self._img_array.shape[:2]
        xi = np.arange(sx)
        yi = np.arange(sy)
        XX, YY = np.meshgrid(xi, yi, indexing="ij")
        circle     = (XX - vox_x) ** 2 + (YY - vox_y) ** 2 <= r ** 2
        group_mask = np.isin(self._label_img[:, :, z], list(self._thresholds.keys()))
        self._erased_mask[:, :, z] |= circle & group_mask
        self._update_plot()

    def _undo(self):
        if self._erase_history:
            self._erased_mask = self._erase_history.pop()
            self._update_plot()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def _masks_for_slice(self, img_slice, label_slice, z, muscle_delta, fat_delta):
        fat_mask       = np.zeros(img_slice.shape, dtype=bool)
        undefined_mask = np.zeros(img_slice.shape, dtype=bool)
        erased_slice   = self._erased_mask[:, :, z]

        for lbl, (muscle_max, fat_min) in self._thresholds.items():
            lbl_mask = (label_slice == lbl) & ~erased_slice
            if not lbl_mask.any():
                continue
            adj_muscle_max = muscle_max + muscle_delta
            if self._components == 2:
                fat_mask |= lbl_mask & (img_slice >= adj_muscle_max)
            else:
                adj_fat_min = (fat_min + fat_delta) if fat_min is not None else adj_muscle_max
                fat_mask       |= lbl_mask & (img_slice >= adj_fat_min)
                undefined_mask |= lbl_mask & (img_slice >= adj_muscle_max) & (img_slice < adj_fat_min)
        return fat_mask, undefined_mask

    def _update_plot(self):
        z            = int(self._valid_slices[self._slice_var.get()])
        muscle_delta = self._muscle_delta_var.get()
        fat_delta    = self._fat_delta_var.get() if self._components == 3 else 0.0

        img_slice   = self._img_array[:, :, z]
        label_slice = self._label_img[:, :, z]
        seg_mask    = np.isin(label_slice, list(self._thresholds.keys()))
        fat_mask, undefined_mask = self._masks_for_slice(
            img_slice, label_slice, z, muscle_delta, fat_delta
        )
        self._cursor_patch = None  # cleared by ax.clear() below

        self._ax.clear()

        vmin = float(np.percentile(img_slice[seg_mask], 1))  if seg_mask.any() else self._vmin
        vmax = float(np.percentile(img_slice[seg_mask], 99)) if seg_mask.any() else self._vmax

        self._ax.imshow(img_slice.T, cmap="gray", origin="lower",
                        vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")

        if undefined_mask.any():
            ov = np.zeros((*img_slice.shape, 4), dtype=np.float32)
            ov[undefined_mask] = [1.0, 0.85, 0.2, 0.5]
            self._ax.imshow(ov.transpose(1, 0, 2), origin="lower", aspect="equal",
                            interpolation="nearest")

        if fat_mask.any():
            ov = np.zeros((*img_slice.shape, 4), dtype=np.float32)
            ov[fat_mask] = [240 / 255, 124 / 255, 44 / 255, 0.5]
            self._ax.imshow(ov.transpose(1, 0, 2), origin="lower", aspect="equal",
                            interpolation="nearest")

        self._ax.set_axis_off()
        self._ax.set_title(f"z = {z}", color=MUTED, fontsize=9, pad=4)
        self._redraw_cursor()

        self._slice_label.configure(text=f"z = {z}")
        sign = "+" if muscle_delta >= 0 else ""
        self._muscle_delta_label.configure(text=f"{sign}{muscle_delta:.2f}")
        if self._components == 3:
            sign = "+" if fat_delta >= 0 else ""
            self._fat_delta_label.configure(text=f"{sign}{fat_delta:.2f}")

    # ------------------------------------------------------------------
    # Reset / Undo / Quit / Accept
    # ------------------------------------------------------------------
    def _confirm_quit(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Quit without saving")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.configure(fg_color=CARD)
        dlg.after(200, lambda: _set_taskbar_icon(dlg))

        ctk.CTkLabel(
            dlg,
            text=(
                "Warning: Your current changes will not be saved.\n\n"
                "If you wish to save your changes, close this dialog\n"
                "and click Accept in the QC window."
            ),
            font=ctk.CTkFont(family=FONT, size=13),
            text_color=TEXT, justify="left", anchor="w",
        ).pack(padx=28, pady=(28, 20), fill="x")

        btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_row.pack(fill="x", padx=28, pady=(0, 24))

        def _do_quit():
            dlg.destroy()
            self.result_muscle_delta = 0.0
            self.result_fat_delta    = 0.0
            self.result_erased_mask  = np.zeros(self._img_array.shape, dtype=bool)
            self.result_quit         = True
            self.grab_release()
            self.destroy()

        ctk.CTkButton(
            btn_row, text="Quit without saving", height=38,
            fg_color=LOGO_RED, hover_color=LOGO_RED_HOVER,
            text_color=PRIMARY_TEXT, corner_radius=19,
            font=ctk.CTkFont(family=FONT, size=13, weight="bold"),
            command=_do_quit,
        ).pack(side="right", padx=(10, 0))

        ctk.CTkButton(
            btn_row, text="Cancel", height=38,
            fg_color="transparent", hover_color="#eeecf4",
            text_color=TEXT, border_color=BORDER, border_width=2,
            corner_radius=19,
            font=ctk.CTkFont(family=FONT, size=13),
            command=dlg.destroy,
        ).pack(side="right")

        dlg.update_idletasks()
        w, h = 420, 220
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        dlg.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _reset(self):
        self._muscle_delta_var.set(0.0)
        if self._components == 3:
            self._fat_delta_var.set(0.0)
        self._update_plot()

    def _accept(self):
        self.result_muscle_delta = self._muscle_delta_var.get()
        self.result_fat_delta    = self._fat_delta_var.get() if self._components == 3 else 0.0
        self.result_erased_mask  = self._erased_mask.copy()
        self.grab_release()
        self.destroy()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
class QCManager:
    def __init__(self):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("musclemap.toolbox.1")
        except Exception:
            pass
        _ensure_ico()
        self._root = ctk.CTk()
        self._root.withdraw()
        self._root.after(200, lambda: _set_taskbar_icon(self._root))
        self.quit_requested = False

    def show(self, img_array, label_img, thresholds, components, anatomy_name=""):
        """
        Returns (muscle_delta, fat_delta, erased_mask).
        erased_mask: bool ndarray, same shape as img_array.
        Sets self.quit_requested=True when user clicks Quit QC.
        """
        win = QCWindow(self._root, img_array, label_img, thresholds, components, anatomy_name)
        self._root.wait_window(win)
        if win.result_quit:
            self.quit_requested = True
        return win.result_muscle_delta, win.result_fat_delta, win.result_erased_mask

    def destroy(self):
        try:
            self._root.destroy()
        except Exception:
            pass
