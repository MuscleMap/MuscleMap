import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pyvista as pv
import vtk

from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent, Qt
from pyvistaqt import QtInteractor

from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from skimage import measure
from skimage.graph import route_through_array
from sklearn.decomposition import PCA


STOMACH_LABEL = None

def derive_seg_and_out_paths(input_path):
    input_path = Path(input_path)
    input_str = str(input_path)

    if input_str.endswith("_dseg.nii.gz"):
        seg_path = input_str
        img_path = input_str.replace("_dseg.nii.gz", ".nii.gz")
        out_path = input_str.replace("_dseg.nii.gz", "_dseg_3comp.nii.gz")
    elif input_str.endswith(".nii.gz"):
        base = input_str[:-7]
        img_path = input_str
        seg_path = base + "_dseg.nii.gz"
        out_path = base + "_dseg_3comp.nii.gz"
    elif input_str.endswith(".nii"):
        base = input_str[:-4]
        img_path = input_str
        seg_path = base + "_dseg.nii.gz"
        out_path = base + "_dseg_3comp.nii.gz"
    else:
        raise ValueError("Input image must be a .nii or .nii.gz file.")
    return img_path, seg_path, out_path

def keep_largest_component(mask):
    cc, n = ndi.label(mask)
    if n == 0:
        raise ValueError("No segmented structure was found.")
    if n == 1:
        return mask
    counts = np.bincount(cc.ravel())
    counts[0] = 0
    return cc == counts.argmax()


def load_mask(seg_path, stomach_label=None):
    img = nib.as_closest_canonical(nib.load(seg_path))
    seg = np.asanyarray(img.dataobj)

    if stomach_label is None:
        mask = seg > 0
    else:
        mask = seg == stomach_label

    mask = keep_largest_component(mask)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_closing(mask, iterations=1)

    return img, mask.astype(bool)


def voxel_sizes_from_affine(affine):
    return nib.affines.voxel_sizes(affine)[:3]


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector encountered.")
    return v / n


def auto_endpoints_from_pca(mask, affine):
    ijk = np.argwhere(mask)
    xyz = nib.affines.apply_affine(affine, ijk)

    pca = PCA(n_components=3)
    scores = pca.fit_transform(xyz)

    start_ijk = tuple(ijk[np.argmin(scores[:, 0])])
    end_ijk = tuple(ijk[np.argmax(scores[:, 0])])

    return start_ijk, end_ijk


def compute_centerline(mask, spacing, start_ijk, end_ijk):
    edt = ndi.distance_transform_edt(mask, sampling=spacing)

    cost = np.full(mask.shape, 1e6, dtype=np.float32)
    cost[mask] = 1.0 / (edt[mask] + 1e-3)

    path, _ = route_through_array(
        cost,
        start=start_ijk,
        end=end_ijk,
        fully_connected=True,
        geometric=True,
    )

    path = np.asarray(path, dtype=np.float32)

    for ax in range(3):
        path[:, ax] = gaussian_filter1d(path[:, ax], sigma=2, mode="nearest")

    return path


def cumulative_arclength(points_xyz):
    if len(points_xyz) < 2:
        return np.array([0.0])
    step = np.linalg.norm(np.diff(points_xyz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def mask_to_mesh(mask, affine):
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    verts_xyz = nib.affines.apply_affine(affine, verts)

    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()

    return pv.PolyData(verts_xyz, faces_pv)


def label_to_mesh(label_map, affine, label_value):
    submask = label_map == label_value
    if np.sum(submask) < 10:
        return None
    try:
        return mask_to_mesh(submask, affine)
    except Exception:
        return None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, img_path, seg_path, out_path):
        super().__init__()
        self.setWindowTitle("3D Stomach Splitter")

        self.img_path = img_path
        self.seg_path = seg_path
        self.out_path = out_path

        self.seg_img, self.mask = load_mask(self.seg_path, STOMACH_LABEL)
        self.affine = self.seg_img.affine
        self.inv_affine = np.linalg.inv(self.affine)
        self.spacing = voxel_sizes_from_affine(self.affine)

        start_ijk, end_ijk = auto_endpoints_from_pca(self.mask, self.affine)
        self.centerline_ijk = compute_centerline(self.mask, self.spacing, start_ijk, end_ijk)
        self.centerline_xyz = nib.affines.apply_affine(self.affine, self.centerline_ijk)
        self.s_cl = cumulative_arclength(self.centerline_xyz)

        self.mesh = mask_to_mesh(self.mask, self.affine)

        self.mask_ijk = np.argwhere(self.mask)
        self.mask_xyz = nib.affines.apply_affine(self.affine, self.mask_ijk)

        self.mode = "rotate"   # rotate, draw_ca, draw_fc
        self.split_done = False
        self.out_labelmap = None
        self.current_view = "Frontal"

        self.lines = {
            "CA": {"color": "blue", "points": [], "view_dir": None},
            "FC": {"color": "red", "points": [], "view_dir": None},
        }

        self._build_ui()
        self._init_scene()
        self._update_workflow_buttons()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout()
        central.setLayout(layout)

        self.plotter = QtInteractor(self)
        self.plotter.interactor.installEventFilter(self)
        layout.addWidget(self.plotter.interactor)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        layout.addLayout(controls)

        self.rotate_button = QtWidgets.QPushButton("Rotate")
        self.rotate_button.clicked.connect(lambda: self.set_mode("rotate"))
        controls.addWidget(self.rotate_button)

        self.instruction_button = QtWidgets.QPushButton("Instructions")
        self.instruction_button.clicked.connect(self.show_instructions)
        controls.addWidget(self.instruction_button)

        self.view_label = QtWidgets.QLabel("View:")
        controls.addWidget(self.view_label)

        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["Frontal", "Sagittal", "Axial"])
        self.view_combo.setCurrentText("Frontal")
        self.view_combo.currentTextChanged.connect(self.set_view_preset)
        controls.addWidget(self.view_combo)

        self.draw_fc_button = QtWidgets.QPushButton("Set FC (2 pt)")
        self.draw_fc_button.clicked.connect(lambda: self.set_mode("draw_fc"))
        controls.addWidget(self.draw_fc_button)

        self.draw_ca_button = QtWidgets.QPushButton("Set CA (2 pt)")
        self.draw_ca_button.clicked.connect(lambda: self.set_mode("draw_ca"))
        controls.addWidget(self.draw_ca_button)

        self.clear_active_button = QtWidgets.QPushButton("Delete active plane")
        self.clear_active_button.clicked.connect(self.clear_active_plane)
        controls.addWidget(self.clear_active_button)

        self.clear_button = QtWidgets.QPushButton("Clear all")
        self.clear_button.clicked.connect(self.clear_all_lines)
        controls.addWidget(self.clear_button)

        self.reverse_checkbox = QtWidgets.QCheckBox("Reverse fundus/antrum label order")
        controls.addWidget(self.reverse_checkbox)

        self.split_button = QtWidgets.QPushButton("Split")
        self.split_button.clicked.connect(self.run_split)
        controls.addWidget(self.split_button)

        self.save_close_button = QtWidgets.QPushButton("Save and Close")
        self.save_close_button.clicked.connect(self.save_and_close)
        controls.addWidget(self.save_close_button)

        self.status_label = QtWidgets.QLabel(
            "Click 2 points in the current view. The plane will then cut through the volume like a knife."
        )
        layout.addWidget(self.status_label)

    def show_instructions(self):
        QtWidgets.QMessageBox.information(
            self,
            "Instructions",
            "Set FC to split fundus and corpus.\n"
            "Set CA to split corpus and antrum."
        )

    def _init_scene(self):
        self.plotter.set_background("white")

        self.plotter.add_mesh(
            self.mesh,
            color="lightgray",
            opacity=0.28,
            smooth_shading=True,
            name="stomach_surface",
            pickable=True,
            reset_camera=False,
        )

        cl_poly = pv.lines_from_points(self.centerline_xyz)
        self.plotter.add_mesh(
            cl_poly,
            color="gold",
            line_width=4,
            name="centerline",
            pickable=False,
            reset_camera=False,
        )

        try:
            self.plotter.enable_trackball_style()
        except Exception:
            pass

        self.set_view_preset("Frontal")

    def set_view_preset(self, view_name, remember=True):
        if remember:
            self.current_view = view_name

        if view_name == "Frontal":
            self.plotter.view_xz(negative=True, render=False)
        elif view_name == "Sagittal":
            self.plotter.view_yz(negative=True, render=False)
        elif view_name == "Axial":
            self.plotter.view_xy(negative=True, render=False)

        self.plotter.render()

    def eventFilter(self, obj, event):
        if obj is self.plotter.interactor and self.mode in ("draw_ca", "draw_fc"):
            et = event.type()

            if et == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    p = self._pick_world_point_from_mouse(event)
                    if p is not None:
                        self._append_point_to_active_line(p)
                    return True
                else:
                    return True

            elif et in (
                QEvent.Wheel,
                QEvent.MouseButtonDblClick,
                QEvent.MouseMove,
                QEvent.MouseButtonRelease,
            ):
                return True

        return super().eventFilter(obj, event)

    def _style_button(self, button, state="neutral", active=False, enabled=True):
        colors = {
            "neutral": ("#e9e9e9", "#222222"),
            "orange": ("#f5b041", "#111111"),
            "green": ("#58d68d", "#111111"),
        }

        bg, fg = colors.get(state, colors["neutral"])
        border = "3px solid #111111" if active else "1px solid #999999"

        if not enabled:
            bg = "#efefef"
            fg = "#9a9a9a"
            border = "1px solid #cccccc"

        button.setEnabled(enabled)
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: {border};
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 600;
                min-height: 30px;
            }}
            """
        )

    def _active_plane_name(self):
        if self.mode == "draw_ca":
            return "CA"
        if self.mode == "draw_fc":
            return "FC"
        return None

    def _has_valid_line(self, name):
        return len(self.lines[name]["points"]) == 2 and self.lines[name]["view_dir"] is not None

    def _update_workflow_buttons(self):
        ca_done = self._has_valid_line("CA")
        fc_done = self._has_valid_line("FC")
        ready_to_split = ca_done and fc_done

        self._style_button(
            self.rotate_button,
            state="neutral",
            active=(self.mode == "rotate"),
            enabled=True,
        )

        self._style_button(
            self.draw_fc_button,
            state="green" if fc_done else "orange",
            active=(self.mode == "draw_fc"),
            enabled=True,
        )

        self._style_button(
            self.draw_ca_button,
            state="green" if ca_done else "orange",
            active=(self.mode == "draw_ca"),
            enabled=True,
        )

        active_name = self._active_plane_name()
        active_exists = active_name is not None and len(self.lines[active_name]["points"]) > 0

        self._style_button(
            self.clear_active_button,
            state="neutral",
            active=False,
            enabled=active_exists,
        )

        self._style_button(
            self.clear_button,
            state="neutral",
            active=False,
            enabled=True,
        )

        if self.split_done:
            self._style_button(
                self.split_button,
                state="green",
                active=False,
                enabled=True,
            )
            self._style_button(
                self.save_close_button,
                state="orange",
                active=False,
                enabled=True,
            )
        else:
            self._style_button(
                self.split_button,
                state="orange" if ready_to_split else "neutral",
                active=False,
                enabled=ready_to_split,
            )
            self._style_button(
                self.save_close_button,
                state="neutral",
                active=False,
                enabled=False,
            )

    def set_mode(self, mode):
        self.mode = mode

        if mode == "rotate":
            try:
                self.plotter.enable_trackball_style()
            except Exception:
                pass
            self.status_label.setText("Mode = ROTATE. Rotate the 3D volume.")

        elif mode == "draw_ca":
            self.set_view_preset(self.current_view, remember=False)
            if self._has_valid_line("CA"):
                self.status_label.setText(
                    "Mode = CA. Plane already exists. Clicking is disabled; use Delete active plane."
                )
            else:
                self.status_label.setText(
                    f"Mode = SET CA. Click 2 points in the {self.current_view.lower()} view; the cutting plane will then be fixed."
                )

        elif mode == "draw_fc":
            self.set_view_preset(self.current_view, remember=False)
            if self._has_valid_line("FC"):
                self.status_label.setText(
                    "Mode = FC. Plane already exists. Clicking is disabled; use Delete active plane."
                )
            else:
                self.status_label.setText(
                    f"Mode = SET FC. Click 2 points in the {self.current_view.lower()} view; the cutting plane will then be fixed."
                )

        self._update_workflow_buttons()

    def _invalidate_split(self):
        self.split_done = False
        self.out_labelmap = None

        for actor_name in [
            "label1_mesh",
            "label2_mesh",
            "label3_mesh",
        ]:
            try:
                self.plotter.remove_actor(actor_name)
            except Exception:
                pass

        self.plotter.render()

    def _pick_world_point_from_mouse(self, event):
        x = int(event.pos().x())
        y = int(self.plotter.interactor.height() - event.pos().y())

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(x, y, 0, self.plotter.renderer)

        if picker.GetCellId() < 0:
            return None

        return np.array(picker.GetPickPosition(), dtype=float)

    def _get_camera_view_dir(self):
        cam = self.plotter.camera
        pos = np.array(cam.GetPosition(), dtype=float)
        focal = np.array(cam.GetFocalPoint(), dtype=float)
        return normalize(focal - pos)

    def _append_point_to_active_line(self, point):
        line_name = "CA" if self.mode == "draw_ca" else "FC"
        pts = self.lines[line_name]["points"]
        p = np.asarray(point, dtype=float)

        if len(pts) >= 2:
            self.status_label.setText(
                f"{line_name} already exists. Use Delete active plane to define it again."
            )
            return

        if len(pts) == 0:
            self.lines[line_name]["view_dir"] = self._get_camera_view_dir()

        if len(pts) == 1:
            if np.linalg.norm(p - pts[0]) < 2.0:
                self.status_label.setText(f"{line_name}: second point is too close to the first point.")
                return

        pts.append(p)

        if len(pts) == 1:
            msg = f"{line_name}: first point set. Now click the second point."
        else:
            msg = (
                f"{line_name}: second point set. Cutting plane has been created. "
                f"Use Delete active plane to redraw it."
            )

        self._invalidate_split()
        self._update_line_visuals(line_name)
        self._update_workflow_buttons()
        self.status_label.setText(msg)

    def _plane_from_two_points_and_view(self, points_xyz, view_dir):
        pts = np.asarray(points_xyz, dtype=float)
        if len(pts) != 2:
            raise ValueError("Exactly 2 points are required for a cutting plane.")

        p0, p1 = pts
        line_vec = p1 - p0
        if np.linalg.norm(line_vec) < 1e-8:
            raise ValueError("The two points are too close together.")

        u = normalize(line_vec)

        w = normalize(view_dir)
        v = w - np.dot(w, u) * u

        if np.linalg.norm(v) < 1e-8:
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ref, u)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            v = ref - np.dot(ref, u) * u

        v = normalize(v)
        n = normalize(np.cross(u, v))
        c = 0.5 * (p0 + p1)

        return {
            "center": c,
            "u": u,
            "v": v,
            "normal": n,
        }

    def _make_plane_preview_mesh(self, plane, points_xyz):
        pts = np.asarray(points_xyz, dtype=float)
        line_len = np.linalg.norm(pts[1] - pts[0])

        bounds = self.mesh.bounds
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        diag = np.sqrt(dx * dx + dy * dy + dz * dz)

        half_u = 1.8 * (0.5 * line_len + 2.0)
        half_v = 1.8 * max(8.0, min(18.0, 0.15 * diag))

        c = plane["center"]
        u = plane["u"]
        v = plane["v"]

        p0 = c - half_u * u - half_v * v
        p1 = c + half_u * u - half_v * v
        p2 = c + half_u * u + half_v * v
        p3 = c - half_u * u + half_v * v

        verts = np.vstack([p0, p1, p2, p3])
        faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)

        return pv.PolyData(verts, faces).triangulate().clean()

    def _signed_distance_to_plane_for_points(self, xyz_points, plane):
        c = plane["center"]
        n = plane["normal"]
        return np.dot(xyz_points - c, n)

    def _find_plane_position_on_centerline(self, plane):
        d = self._signed_distance_to_plane_for_points(self.centerline_xyz, plane)

        idx = int(np.argmin(np.abs(d)))

        sign_change_idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        if len(sign_change_idx) > 0:
            idx = int(sign_change_idx[np.argmin(np.abs(sign_change_idx - idx))])

        return idx, self.s_cl[idx]

    def _connected_mask_near_centerline_band(self, region_mask, target_s, band_mm=25.0):
        cl_band = np.abs(self.s_cl - target_s) <= band_mm
        if not np.any(cl_band):
            cl_band[np.argmin(np.abs(self.s_cl - target_s))] = True

        centerline_band_pts = self.centerline_xyz[cl_band]

        if len(centerline_band_pts) == 0:
            return region_mask

        band_hits = np.zeros(self.mask.shape, dtype=bool)

        for p in centerline_band_pts:
            d = np.linalg.norm(self.mask_xyz - p, axis=1)
            near = d <= max(self.spacing) * 2.5
            if np.any(near):
                vox = self.mask_ijk[near]
                band_hits[vox[:, 0], vox[:, 1], vox[:, 2]] = True

        seeds = region_mask & band_hits
        if not np.any(seeds):
            return region_mask

        cc, n = ndi.label(region_mask)
        if n == 0:
            return region_mask

        seed_labels = np.unique(cc[seeds])
        seed_labels = seed_labels[seed_labels > 0]

        keep = np.isin(cc, seed_labels)
        return keep

    def _split_mask_with_planes(self):
        ca_points = self.lines["CA"]["points"]
        fc_points = self.lines["FC"]["points"]

        ca_view = self.lines["CA"]["view_dir"]
        fc_view = self.lines["FC"]["view_dir"]

        if len(ca_points) != 2 or ca_view is None:
            raise ValueError("CA does not yet have a complete plane.")
        if len(fc_points) != 2 or fc_view is None:
            raise ValueError("FC does not yet have a complete plane.")

        plane1 = self._plane_from_two_points_and_view(ca_points, ca_view)
        plane2 = self._plane_from_two_points_and_view(fc_points, fc_view)

        idx1, s1 = self._find_plane_position_on_centerline(plane1)
        idx2, s2 = self._find_plane_position_on_centerline(plane2)

        if np.isclose(s1, s2):
            raise ValueError("The two planes are located almost at the same position along the centerline.")

        if s1 <= s2:
            first_plane = plane1
            second_plane = plane2
            s_first = s1
            s_second = s2
        else:
            first_plane = plane2
            second_plane = plane1
            s_first = s2
            s_second = s1

        d_first = self._signed_distance_to_plane_for_points(self.mask_xyz, first_plane)
        d_second = self._signed_distance_to_plane_for_points(self.mask_xyz, second_plane)

        mid_s = 0.5 * (s_first + s_second)
        center_idx = np.argmin(np.abs(self.s_cl - mid_s))
        center_pt = self.centerline_xyz[center_idx:center_idx + 1]

        d1_mid = self._signed_distance_to_plane_for_points(center_pt, first_plane)[0]
        d2_mid = self._signed_distance_to_plane_for_points(center_pt, second_plane)[0]

        if d1_mid >= 0 and d2_mid >= 0:
            middle_mask_xyz = (d_first >= 0) & (d_second >= 0)
        elif d1_mid >= 0 and d2_mid < 0:
            middle_mask_xyz = (d_first >= 0) & (d_second < 0)
        elif d1_mid < 0 and d2_mid >= 0:
            middle_mask_xyz = (d_first < 0) & (d_second >= 0)
        else:
            middle_mask_xyz = (d_first < 0) & (d_second < 0)

        mid_region = np.zeros(self.mask.shape, dtype=bool)
        vox_mid = self.mask_ijk[middle_mask_xyz]
        mid_region[vox_mid[:, 0], vox_mid[:, 1], vox_mid[:, 2]] = True
        mid_region &= self.mask

        mid_region = self._connected_mask_near_centerline_band(mid_region, mid_s, band_mm=25.0)

        remaining = self.mask & (~mid_region)
        cc, n = ndi.label(remaining)

        if n < 2:
            raise ValueError(
                "The two straight planes do not produce 3 clean compartments. "
                "Try placing the points more from edge to edge."
            )

        counts = np.bincount(cc.ravel())
        counts[0] = 0
        comp_ids = np.argsort(counts)[::-1]
        comp_ids = [cid for cid in comp_ids if counts[cid] > 0][:2]

        if len(comp_ids) < 2:
            raise ValueError("Could not find two outer compartments.")

        comp_a = cc == comp_ids[0]
        comp_b = cc == comp_ids[1]

        start_band_s = self.s_cl[0] + 0.10 * (self.s_cl[-1] - self.s_cl[0])
        end_band_s = self.s_cl[-1] - 0.10 * (self.s_cl[-1] - self.s_cl[0])

        prox_mask_a = self._connected_mask_near_centerline_band(comp_a, start_band_s, band_mm=25.0)
        prox_mask_b = self._connected_mask_near_centerline_band(comp_b, start_band_s, band_mm=25.0)

        hit_a_start = np.any(prox_mask_a & comp_a)
        hit_b_start = np.any(prox_mask_b & comp_b)

        if hit_a_start and not hit_b_start:
            first_region = comp_a
            third_region = comp_b
        elif hit_b_start and not hit_a_start:
            first_region = comp_b
            third_region = comp_a
        else:
            start_pt = self.centerline_xyz[0]
            end_pt = self.centerline_xyz[-1]

            comp_a_xyz = nib.affines.apply_affine(self.affine, np.argwhere(comp_a))
            comp_b_xyz = nib.affines.apply_affine(self.affine, np.argwhere(comp_b))

            da_start = np.min(np.linalg.norm(comp_a_xyz - start_pt, axis=1)) if len(comp_a_xyz) else np.inf
            db_start = np.min(np.linalg.norm(comp_b_xyz - start_pt, axis=1)) if len(comp_b_xyz) else np.inf

            if da_start <= db_start:
                first_region = comp_a
                third_region = comp_b
            else:
                first_region = comp_b
                third_region = comp_a

        out = np.zeros(self.mask.shape, dtype=np.uint8)

        if self.reverse_checkbox.isChecked():
            out[first_region] = 3
            out[mid_region] = 2
            out[third_region] = 1
        else:
            out[first_region] = 1
            out[mid_region] = 2
            out[third_region] = 3

        out[~self.mask] = 0

        present = sorted([x for x in np.unique(out) if x > 0])
        if present != [1, 2, 3]:
            raise ValueError(
                f"Split did not produce 3 valid labels, but {present}. "
                f"Try positioning the cutting planes more carefully."
            )

        return out.astype(np.uint8)

    def _update_line_visuals(self, line_name):
        data = self.lines[line_name]
        color = data["color"]
        pts = np.asarray(data["points"], dtype=float)
        view_dir = data["view_dir"]
        prefix = line_name.lower()

        for actor_name in [
            f"{prefix}_points",
            f"{prefix}_line",
            f"{prefix}_plane_preview",
        ]:
            try:
                self.plotter.remove_actor(actor_name)
            except Exception:
                pass

        if len(pts) > 0:
            point_cloud = pv.PolyData(pts)
            self.plotter.add_mesh(
                point_cloud,
                color=color,
                point_size=12,
                render_points_as_spheres=True,
                name=f"{prefix}_points",
                pickable=False,
                reset_camera=False,
            )

        if len(pts) == 2:
            line = pv.Line(pts[0], pts[1], resolution=1)
            self.plotter.add_mesh(
                line,
                color=color,
                line_width=6,
                name=f"{prefix}_line",
                pickable=False,
                reset_camera=False,
            )

            if view_dir is not None:
                plane = self._plane_from_two_points_and_view(pts, view_dir)
                plane_mesh = self._make_plane_preview_mesh(plane, pts)

                self.plotter.add_mesh(
                    plane_mesh,
                    color=color,
                    opacity=0.18,
                    name=f"{prefix}_plane_preview",
                    pickable=False,
                    reset_camera=False,
                )

        self.plotter.render()

    def clear_active_plane(self):
        line_name = self._active_plane_name()
        if line_name is None:
            self.status_label.setText("First select FC or CA to delete a plane.")
            return

        self._invalidate_split()

        self.lines[line_name]["points"] = []
        self.lines[line_name]["view_dir"] = None

        prefix = line_name.lower()
        for actor_name in [
            f"{prefix}_points",
            f"{prefix}_line",
            f"{prefix}_plane_preview",
        ]:
            try:
                self.plotter.remove_actor(actor_name)
            except Exception:
                pass

        self.plotter.render()
        self._update_workflow_buttons()
        self.status_label.setText(f"{line_name} deleted. You can now set 2 new points.")

    def clear_all_lines(self):
        self._invalidate_split()

        for line_name in ["CA", "FC"]:
            self.lines[line_name]["points"] = []
            self.lines[line_name]["view_dir"] = None

            prefix = line_name.lower()
            for actor_name in [
                f"{prefix}_points",
                f"{prefix}_line",
                f"{prefix}_plane_preview",
            ]:
                try:
                    self.plotter.remove_actor(actor_name)
                except Exception:
                    pass

        self.plotter.render()
        self.status_label.setText("All points and planes cleared.")
        self._update_workflow_buttons()

    def run_split(self):
        try:
            out = self._split_mask_with_planes()

            self.out_labelmap = out
            self.split_done = True

            self._show_result_mesh(out)
            self._update_workflow_buttons()

            total = np.sum(out > 0)
            frac1 = 100 * np.sum(out == 1) / total if total > 0 else 0
            frac2 = 100 * np.sum(out == 2) / total if total > 0 else 0
            frac3 = 100 * np.sum(out == 3) / total if total > 0 else 0

            self.status_label.setText(
                f"Split complete | Label 1: {frac1:.1f}% | Label 2: {frac2:.1f}% | Label 3: {frac3:.1f}%"
            )

        except Exception as e:
            self.split_done = False
            self.out_labelmap = None
            self._update_workflow_buttons()
            self.status_label.setText(f"Split error: {e}")

    def _show_result_mesh(self, out):
        for actor_name in [
            "label1_mesh",
            "label2_mesh",
            "label3_mesh",
        ]:
            try:
                self.plotter.remove_actor(actor_name)
            except Exception:
                pass

        label1_mesh = label_to_mesh(out, self.affine, 1)
        label2_mesh = label_to_mesh(out, self.affine, 2)
        label3_mesh = label_to_mesh(out, self.affine, 3)

        if label1_mesh is not None:
            self.plotter.add_mesh(
                label1_mesh,
                color="red",
                opacity=0.90,
                smooth_shading=True,
                name="label1_mesh",
                pickable=False,
                reset_camera=False,
            )

        if label2_mesh is not None:
            self.plotter.add_mesh(
                label2_mesh,
                color="green",
                opacity=0.90,
                smooth_shading=True,
                name="label2_mesh",
                pickable=False,
                reset_camera=False,
            )

        if label3_mesh is not None:
            self.plotter.add_mesh(
                label3_mesh,
                color="blue",
                opacity=0.90,
                smooth_shading=True,
                name="label3_mesh",
                pickable=False,
                reset_camera=False,
            )

        self.plotter.render()

    def save_and_close(self):
        try:
            if not self.split_done or self.out_labelmap is None:
                self.run_split()

            if self.out_labelmap is None:
                raise ValueError("There is no split available to save.")

            header = self.seg_img.header.copy()
            header.set_data_dtype(np.uint8)
            out_img = nib.Nifti1Image(self.out_labelmap.astype(np.uint8), self.affine, header)
            nib.save(out_img, self.out_path)

            self.status_label.setText(f"Saved to {self.out_path}")
            QtWidgets.QApplication.processEvents()

            self.close()
            QtWidgets.QApplication.quit()

        except Exception as e:
            self.status_label.setText(f"Save error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D stomach splitter. Give the image path; segmentation path is derived automatically."
    )
    parser.add_argument(
        "-i", '--input_image',
        required = True,
        type=str,
        help="Path to the input image (.nii or .nii.gz). Segmentation is expected as <image>_dseg.nii.gz",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    img_path, seg_path, out_path = derive_seg_and_out_paths(args.input_image)

    if not Path(img_path).exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")
    if not Path(seg_path).exists():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    pv.set_plot_theme("document")
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(img_path, seg_path, out_path)
    window.resize(1180, 760)
    window.show()
    print('Output in components saved to input folder')
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()