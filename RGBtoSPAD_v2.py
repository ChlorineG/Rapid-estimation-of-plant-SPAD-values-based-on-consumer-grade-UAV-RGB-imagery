import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageFont
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from datetime import datetime
import platform
import sys
import warnings
import threading
from collections import OrderedDict

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calculate_rgb_with_filter(roi, use_green_filter=False, green_threshold=50):
    if not use_green_filter:
        rgb_mean = cv2.mean(roi)[:3]
        return rgb_mean[2], rgb_mean[1], rgb_mean[0]

    roi_reshaped = roi.reshape(-1, 3)
    green_mask = (roi_reshaped[:, 1] - np.maximum(roi_reshaped[:, 0], roi_reshaped[:, 2])) > green_threshold

    if np.sum(green_mask) == 0:
        rgb_mean = cv2.mean(roi)[:3]
        return rgb_mean[2], rgb_mean[1], rgb_mean[0]

    green_pixels = roi_reshaped[green_mask]
    r_mean = np.mean(green_pixels[:, 2])
    g_mean = np.mean(green_pixels[:, 1])
    b_mean = np.mean(green_pixels[:, 0])

    return r_mean, g_mean, b_mean


class LRUCache:
    def __init__(self, max_size=15):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


class SPADPredictor:
    def __init__(self):
        self.models = {'linear': None, 'elastic': None}
        self.model_params = {
            'linear': {},
            'elastic': {'alpha': 1.0, 'l1_ratio': 0.5}
        }
        self.active_model = 'linear'
        self.model_metrics = {
            'linear': {'r2': None, 'mae': None, 'coef': None, 'intercept': None},
            'elastic': {'r2': None, 'mae': None, 'coef': None, 'intercept': None}
        }
        self.calibration_points = []
        self.is_trained = False
        self.feature_names = ['R', 'G', 'B']

    def add_calibration_point(self, r, g, b, spad, region):
        self.calibration_points.append({
            'R': r, 'G': g, 'B': b,
            'SPAD': spad, 'region': region,
            'index': len(self.calibration_points) + 1
        })
        self.is_trained = False

    def remove_calibration_point(self, index):
        if 0 <= index < len(self.calibration_points):
            self.calibration_points.pop(index)
            for i, p in enumerate(self.calibration_points):
                p['index'] = i + 1
            self.is_trained = False
            self.models = {'linear': None, 'elastic': None}

    def train(self):
        if len(self.calibration_points) < 3:
            raise ValueError(f"至少需要3个标定点，当前只有{len(self.calibration_points)}个")

        df = pd.DataFrame(self.calibration_points)
        X = df[["R", "G", "B"]]
        y = df["SPAD"]

        self.models['linear'] = LinearRegression()
        self.models['linear'].fit(X, y)
        y_pred_linear = self.models['linear'].predict(X)
        self.model_metrics['linear'] = {
            'r2': r2_score(y, y_pred_linear),
            'mae': mean_absolute_error(y, y_pred_linear),
            'coef': self.models['linear'].coef_,
            'intercept': self.models['linear'].intercept_
        }

        params = self.model_params['elastic']
        self.models['elastic'] = ElasticNet(
            alpha=params['alpha'],
            l1_ratio=params['l1_ratio'],
            max_iter=10000
        )
        self.models['elastic'].fit(X, y)
        y_pred_elastic = self.models['elastic'].predict(X)
        self.model_metrics['elastic'] = {
            'r2': r2_score(y, y_pred_elastic),
            'mae': mean_absolute_error(y, y_pred_elastic),
            'coef': self.models['elastic'].coef_,
            'intercept': self.models['elastic'].intercept_
        }

        self.is_trained = True

    def get_equation_string(self, model_type=None):
        if model_type is None:
            model_type = self.active_model
        metrics = self.model_metrics[model_type]
        a, b, c = metrics['coef']
        d = metrics['intercept']
        return f"SPAD = {a:.4f} × R + {b:.4f} × G + {c:.4f} × B + {d:.4f}"

    def get_metrics(self, model_type=None):
        if model_type is None:
            model_type = self.active_model
        return self.model_metrics[model_type]

    def predict(self, r, g, b):
        if not self.is_trained:
            raise ValueError("模型未训练")
        input_data = pd.DataFrame([[r, g, b]], columns=self.feature_names)
        return self.models[self.active_model].predict(input_data)[0]

    def set_active_model(self, model_type):
        if model_type in self.models and self.models[model_type] is not None:
            self.active_model = model_type

    def set_elastic_params(self, alpha, l1_ratio):
        self.model_params['elastic'] = {'alpha': alpha, 'l1_ratio': l1_ratio}
        if self.is_trained and len(self.calibration_points) >= 3:
            self.train()


class ImageViewer:
    def __init__(self, canvas, on_select=None, enable_drag=True):
        self.canvas = canvas
        self.on_select = on_select
        self.enable_drag = enable_drag
        self.original_image = None
        self.rotated_image = None
        self.display_image = None
        self.photo = None
        self.scale = 1.0
        self.rotation = 0
        self.target_rotation = 0
        self.offset_x = self.offset_y = 0
        self.select_start = None
        self.select_rect = None
        self.drag_start_x = self.drag_start_y = 0
        self.drag_start_offset_x = self.drag_start_offset_y = 0
        self.is_dragging = False
        self._rotation_update_pending = False
        self._rotation_cache = LRUCache(max_size=12)
        self._preload_angles = [-180, -135, -90, -45, 0, 45, 90, 135, 180]

        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.show_menu)
        self.canvas.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        if self.original_image:
            self.fit_to_window()

    def set_image(self, img):
        self.original_image = img
        self.rotation = 0
        self.target_rotation = 0
        self.rotated_image = img
        self.scale = 1.0
        self.offset_x = self.offset_y = 0
        self._rotation_cache.clear()
        self.update_display()
        self.canvas.after(100, self.fit_to_window)
        self._preload_rotations()

    def _preload_rotations(self):
        if self.original_image is None:
            return

        def preload_task():
            for angle in self._preload_angles:
                cache_key = (id(self.original_image), angle)
                if self._rotation_cache.get(cache_key) is None:
                    if angle != 0:
                        rotated = self.original_image.rotate(-angle, expand=True,
                                                             resample=Image.Resampling.BICUBIC)
                    else:
                        rotated = self.original_image
                    self._rotation_cache.put(cache_key, rotated)

        threading.Thread(target=preload_task, daemon=True).start()

    def set_rotation(self, angle):
        angle = round(angle)

        if self.target_rotation == angle:
            return

        self.target_rotation = angle

        if self._rotation_update_pending:
            return

        self._rotation_update_pending = True

        def process_rotation():
            self._rotation_update_pending = False
            if self.target_rotation == self.rotation:
                return

            self.rotation = self.target_rotation

            cache_key = (id(self.original_image), self.rotation)
            cached = self._rotation_cache.get(cache_key)

            if cached is not None:
                self.rotated_image = cached
                self.center()
                self.update_display()
                return

            def rotate_task():
                if self.rotation != 0:
                    new_rotated = self.original_image.rotate(-self.rotation, expand=True,
                                                             resample=Image.Resampling.BICUBIC)
                else:
                    new_rotated = self.original_image

                self._rotation_cache.put(cache_key, new_rotated)
                self.canvas.after(0, lambda: self._update_rotated_image(new_rotated))

            threading.Thread(target=rotate_task, daemon=True).start()

        self.canvas.after(10, process_rotation)

    def _update_rotated_image(self, new_rotated):
        self.rotated_image = new_rotated
        self.center()
        self.update_display()

    def get_rotated_image(self):
        if self.rotated_image is not None:
            return self.rotated_image
        return self.original_image

    def get_original_image(self):
        return self.original_image

    def center(self):
        if self.original_image:
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w > 1 and h > 1:
                rotated = self.get_rotated_image()
                dw, dh = int(rotated.width * self.scale), int(rotated.height * self.scale)
                self.offset_x, self.offset_y = (w - dw) // 2, (h - dh) // 2

    def transform(self):
        if not self.original_image:
            return None
        img = self.get_rotated_image()
        if self.scale != 1.0:
            img = img.resize((int(img.width * self.scale), int(img.height * self.scale)),
                             Image.Resampling.LANCZOS)
        return img

    def update_display(self):
        if not self.original_image:
            return
        self.display_image = self.transform()
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("display_image")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW,
                                 image=self.photo, tags="display_image")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.tag_lower("display_image")

    def to_image(self, cx, cy):
        img_x = (cx - self.offset_x) / self.scale
        img_y = (cy - self.offset_y) / self.scale

        if self.rotation != 0:
            rad = math.radians(-self.rotation)
            cos_t = math.cos(rad)
            sin_t = math.sin(rad)

            rotated = self.get_rotated_image()
            rot_w, rot_h = rotated.width, rotated.height
            orig_w, orig_h = self.original_image.width, self.original_image.height
            center_x = rot_w / 2
            center_y = rot_h / 2
            orig_center_x = orig_w / 2
            orig_center_y = orig_h / 2

            tx = img_x - center_x
            ty = img_y - center_y
            orig_x = tx * cos_t - ty * sin_t + orig_center_x
            orig_y = tx * sin_t + ty * cos_t + orig_center_y

            return int(orig_x), int(orig_y)

        return int(img_x), int(img_y)

    def to_canvas(self, ix, iy):
        if self.rotation != 0:
            rad = math.radians(self.rotation)
            cos_t = math.cos(rad)
            sin_t = math.sin(rad)

            orig_w, orig_h = self.original_image.width, self.original_image.height
            orig_center_x = orig_w / 2
            orig_center_y = orig_h / 2

            rotated = self.get_rotated_image()
            rot_w, rot_h = rotated.width, rotated.height
            rot_center_x = rot_w / 2
            rot_center_y = rot_h / 2

            tx = ix - orig_center_x
            ty = iy - orig_center_y
            rot_x = tx * cos_t - ty * sin_t + rot_center_x
            rot_y = tx * sin_t + ty * cos_t + rot_center_y
        else:
            rot_x, rot_y = ix, iy

        canvas_x = rot_x * self.scale + self.offset_x
        canvas_y = rot_y * self.scale + self.offset_y
        return canvas_x, canvas_y

    def on_press(self, e):
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        if self.enable_drag:
            self.drag_start_x = cx
            self.drag_start_y = cy
            self.drag_start_offset_x = self.offset_x
            self.drag_start_offset_y = self.offset_y
            self.is_dragging = True
        self.select_start = self.to_image(cx, cy)

    def on_motion(self, e):
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)

        if self.enable_drag and self.is_dragging:
            self.offset_x = self.drag_start_offset_x + (cx - self.drag_start_x)
            self.offset_y = self.drag_start_offset_y + (cy - self.drag_start_y)
            self.update_display()

        if self.select_start and self.on_select:
            if self.select_rect:
                self.canvas.delete(self.select_rect)

            ex, ey = self.to_image(cx, cy)
            x1, y1 = self.select_start
            x2, y2 = ex, ey
            l, r = min(x1, x2), max(x1, x2)
            t, b = min(y1, y2), max(y1, y2)

            if l != r and t != b:
                cl, ct = self.to_canvas(l, t)
                cr, cb = self.to_canvas(r, b)
                self.select_rect = self.canvas.create_rectangle(cl, ct, cr, cb,
                                                                outline='yellow', width=2, dash=(5, 5))

    def on_release(self, e):
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        self.is_dragging = False

        if self.select_start and self.on_select:
            if self.select_rect:
                self.canvas.delete(self.select_rect)
                self.select_rect = None

            ex, ey = self.to_image(cx, cy)
            x1, y1 = self.select_start
            x2, y2 = ex, ey
            l, r = min(x1, x2), max(x1, x2)
            t, b = min(y1, y2), max(y1, y2)

            if l != r and t != b and (r - l) > 5 and (b - t) > 5:
                self.on_select(l, t, r, b)

            self.select_start = None

    def zoom(self, delta, x=None, y=None):
        old = self.scale
        self.scale = max(0.1, min(5.0, self.scale * (1.1 if delta > 0 else 0.9)))
        if x and y and old != self.scale:
            self.offset_x = x - (x - self.offset_x) * (self.scale / old)
            self.offset_y = y - (y - self.offset_y) * (self.scale / old)
        else:
            self.center()
        self.update_display()

    def on_mousewheel(self, e):
        self.zoom(-1 if e.delta < 0 else 1, self.canvas.canvasx(e.x), self.canvas.canvasy(e.y))

    def show_menu(self, e):
        m = tk.Menu(self.canvas, tearoff=0)
        m.add_command(label="放大", command=lambda: self.zoom(1))
        m.add_command(label="缩小", command=lambda: self.zoom(-1))
        m.add_separator()
        m.add_command(label="将图片按合适比例缩放", command=self.fit_to_window)
        m.post(e.x_root, e.y_root)

    def fit_to_window(self):
        if self.original_image:
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w > 1 and h > 1:
                rotated = self.get_rotated_image()
                self.scale = min(w / rotated.width, h / rotated.height)
                self.center()
                self.update_display()


class ThresholdDialog:
    def __init__(self, parent, current_threshold, callback):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("设置绿色过滤阈值")
        self.dialog.geometry("350x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.callback = callback

        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (350 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (150 // 2)
        self.dialog.geometry(f"350x150+{x}+{y}")

        ttk.Label(self.dialog, text="绿色过滤阈值", font=("Arial", 11, "bold")).pack(pady=10)
        ttk.Label(self.dialog, text="绿色判定: G - max(R, B) > 阈值", font=("Arial", 9), foreground="gray").pack()

        frame = ttk.Frame(self.dialog)
        frame.pack(pady=15)

        ttk.Label(frame, text="阈值 (0-100):").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.StringVar(value=str(current_threshold))
        entry = ttk.Entry(frame, textvariable=self.threshold_var, width=10, font=("Arial", 11))
        entry.pack(side=tk.LEFT, padx=5)
        entry.focus()
        entry.select_range(0, tk.END)

        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(pady=10)

        def confirm():
            try:
                value = float(self.threshold_var.get())
                if 0 <= value <= 100:
                    self.callback(value)
                    self.dialog.destroy()
                else:
                    messagebox.showerror("错误", "阈值必须在0-100之间！")
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字！")

        ttk.Button(btn_frame, text="确定", command=confirm, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT, padx=10)

        entry.bind('<Return>', lambda event: confirm())


class ElasticNetSettingsDialog:
    def __init__(self, parent, current_alpha, current_l1_ratio, callback):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Elastic Net 参数设置")
        self.dialog.geometry("400x220")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.callback = callback

        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (220 // 2)
        self.dialog.geometry(f"400x220+{x}+{y}")

        ttk.Label(self.dialog, text="Elastic Net 参数设置",
                  font=("Arial", 11, "bold")).pack(pady=10)
        ttk.Label(self.dialog, text="调整参数后模型将自动重新训练",
                  font=("Arial", 9), foreground="gray").pack()

        frame1 = ttk.Frame(self.dialog)
        frame1.pack(pady=8)
        ttk.Label(frame1, text="Alpha (正则化强度, 0.01-10):").pack(side=tk.LEFT, padx=5)
        self.alpha_var = tk.StringVar(value=str(current_alpha))
        ttk.Entry(frame1, textvariable=self.alpha_var, width=10).pack(side=tk.LEFT, padx=5)

        frame2 = ttk.Frame(self.dialog)
        frame2.pack(pady=8)
        ttk.Label(frame2, text="L1 Ratio (L1占比, 0-1):").pack(side=tk.LEFT, padx=5)
        self.l1_var = tk.StringVar(value=str(current_l1_ratio))
        ttk.Entry(frame2, textvariable=self.l1_var, width=10).pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(pady=15)

        def confirm():
            try:
                alpha = float(self.alpha_var.get())
                l1_ratio = float(self.l1_var.get())
                if not (0.01 <= alpha <= 10):
                    messagebox.showerror("错误", "Alpha必须在0.01-10之间！")
                    return
                if not (0 <= l1_ratio <= 1):
                    messagebox.showerror("错误", "L1 Ratio必须在0-1之间！")
                    return
                self.callback(alpha, l1_ratio)
                self.dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字！")

        ttk.Button(btn_frame, text="确定", command=confirm, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT, padx=10)


class CalibrationDialog:
    def __init__(self, parent, image, image_cv, callback, use_green_filter=False, green_threshold=50):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("校准模式 - 添加标定点")
        self.dialog.geometry("1500x900")
        self.dialog.minsize(1200, 700)

        self.callback = callback
        self.image = image
        self.image_cv = image_cv
        self.use_green_filter = use_green_filter
        self.green_threshold = green_threshold
        self.predictor = SPADPredictor()

        self.setup_ui()
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.after(100, lambda: self.viewer.fit_to_window())

    def setup_ui(self):
        main = ttk.Frame(self.dialog)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctrl = ttk.Frame(main)
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        ttk.Label(ctrl, text="校准模式：在图片上拖动鼠标选择标定区域，输入SPAD值", foreground="red").pack(side=tk.LEFT,
                                                                                                        padx=5)
        self.calib_status = ttk.Label(ctrl, text="标定点数量: 0", foreground="blue")
        self.calib_status.pack(side=tk.LEFT, padx=20)
        ttk.Button(ctrl, text="完成校准并拟合", command=self.finish, width=15).pack(side=tk.RIGHT, padx=5)

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(paned, text="图片区域 (鼠标滚轮缩放，拖动选择区域)", padding=5)
        paned.add(left_frame, weight=5)

        self.canvas_frame = ttk.Frame(left_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(self.canvas_frame, bg='gray', highlightthickness=0)
        h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.image_canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.image_canvas.grid(row=0, column=0, sticky='nsew')
        h_scroll.grid(row=1, column=0, sticky='ew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        right_frame = ttk.LabelFrame(paned, text="标定点列表", padding=5)
        paned.add(right_frame, weight=1)

        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        cols = ('编号', 'X1', 'Y1', 'X2', 'Y2', 'R', 'G', 'B', 'SPAD')
        self.tree = ttk.Treeview(list_frame, columns=cols, show='headings', height=18)
        widths = {'编号': 45, 'X1': 50, 'Y1': 50, 'X2': 50, 'Y2': 50, 'R': 50, 'G': 50, 'B': 50, 'SPAD': 55}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths.get(c, 55), anchor='center')

        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="删除选中标定点", command=self.delete_selected, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="删除全部标定点", command=self.clear_all, width=15).pack(side=tk.LEFT, padx=5)

        self.tree.bind('<<TreeviewSelect>>', self.on_select)

        self.viewer = ImageViewer(self.image_canvas, self.on_region_selected, enable_drag=False)
        self.viewer.set_image(self.image)

    def on_region_selected(self, x1, y1, x2, y2):
        roi = self.image_cv[y1:y2, x1:x2]
        r, g, b = calculate_rgb_with_filter(roi, self.use_green_filter, self.green_threshold)
        self.show_spad_dialog(x1, y1, x2, y2, r, g, b)

    def show_spad_dialog(self, x1, y1, x2, y2, r, g, b):
        dialog = tk.Toplevel(self.dialog)
        dialog.title("输入SPAD值")
        dialog.geometry("400x300")
        dialog.resizable(False, False)
        dialog.transient(self.dialog)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (300 // 2)
        dialog.geometry(f"400x300+{x}+{y}")

        ttk.Label(dialog, text="标定点信息", font=("Arial", 12, "bold")).pack(pady=10)

        f = ttk.Frame(dialog)
        f.pack(pady=5)
        ttk.Label(f, text=f"区域位置: ({x1}, {y1}) → ({x2}, {y2})").pack()
        ttk.Label(f, text=f"区域尺寸: {x2 - x1} × {y2 - y1} 像素").pack(pady=3)

        filter_text = " (已应用绿色过滤)" if self.use_green_filter else ""
        rf = ttk.LabelFrame(dialog, text=f"RGB平均值{filter_text}", padding=5)
        rf.pack(pady=5, padx=20, fill=tk.X)
        ttk.Label(rf, text=f"R = {r:.1f}    G = {g:.1f}    B = {b:.1f}",
                  font=("Arial", 10, "bold"), foreground="blue").pack()

        inf = ttk.Frame(dialog)
        inf.pack(pady=10)
        ttk.Label(inf, text="真实SPAD值:").pack(side=tk.LEFT, padx=5)
        spad_var = tk.StringVar()
        spad_entry = ttk.Entry(inf, textvariable=spad_var, width=10, font=("Arial", 12))
        spad_entry.pack(side=tk.LEFT, padx=5)
        spad_entry.focus()

        def confirm():
            try:
                spad = float(spad_var.get())
                self.add_calibration_point(x1, y1, x2, y2, r, g, b, spad)
                dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效数字！")

        bf = ttk.Frame(dialog)
        bf.pack(pady=15)
        ttk.Button(bf, text="确定", command=confirm, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Button(bf, text="取消", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=10)
        spad_entry.bind('<Return>', lambda event: confirm())

    def add_calibration_point(self, x1, y1, x2, y2, r, g, b, spad):
        self.predictor.add_calibration_point(r, g, b, spad, (x1, y1, x2, y2))
        self.update_display()

        idx = len(self.predictor.calibration_points)
        self.tree.insert('', 'end', values=(idx, x1, y1, x2, y2, f"{r:.1f}", f"{g:.1f}", f"{b:.1f}", spad))
        self.calib_status.config(text=f"标定点数量: {idx}")

    def update_display(self):
        self.image_canvas.delete("calibration")
        for p in self.predictor.calibration_points:
            x1, y1, x2, y2 = p['region']
            cx1, cy1 = self.viewer.to_canvas(x1, y1)
            cx2, cy2 = self.viewer.to_canvas(x2, y2)
            self.image_canvas.create_rectangle(cx1, cy1, cx2, cy2, outline='red', width=2, tags="calibration")
            cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
            self.image_canvas.create_text(cx, cy, text=f"{p['SPAD']}", fill='red', font=('Arial', 10, 'bold'),
                                          tags="calibration")

    def on_select(self, e):
        sel = self.tree.selection()
        if sel:
            idx = self.tree.index(sel[0])
            self.highlight_point(idx)

    def highlight_point(self, idx):
        self.image_canvas.delete("highlight")
        if 0 <= idx < len(self.predictor.calibration_points):
            x1, y1, x2, y2 = self.predictor.calibration_points[idx]['region']
            cx1, cy1 = self.viewer.to_canvas(x1, y1)
            cx2, cy2 = self.viewer.to_canvas(x2, y2)
            self.image_canvas.create_rectangle(cx1 - 2, cy1 - 2, cx2 + 2, cy2 + 2, outline='orange', width=3,
                                               tags="highlight")

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("提示", "请先选择要删除的标定点")
            return

        idx = self.tree.index(sel[0])
        self.predictor.remove_calibration_point(idx)

        for item in self.tree.get_children():
            self.tree.delete(item)
        for p in self.predictor.calibration_points:
            x1, y1, x2, y2 = p['region']
            self.tree.insert('', 'end', values=(p['index'], x1, y1, x2, y2,
                                                f"{p['R']:.1f}", f"{p['G']:.1f}", f"{p['B']:.1f}", p['SPAD']))

        self.update_display()
        self.calib_status.config(text=f"标定点数量: {len(self.predictor.calibration_points)}")
        self.image_canvas.delete("highlight")

    def clear_all(self):
        if messagebox.askyesno("确认", "确定要删除所有标定点吗？"):
            self.predictor.calibration_points = []
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.update_display()
            self.calib_status.config(text="标定点数量: 0")
            self.image_canvas.delete("highlight")

    def finish(self):
        if len(self.predictor.calibration_points) < 3:
            messagebox.showwarning("警告", f"至少需要3个标定点，当前只有{len(self.predictor.calibration_points)}个")
            return

        self.predictor.train()
        self.callback(self.predictor)
        self.dialog.destroy()


class RegionSelectDialog:
    def __init__(self, parent, image, image_cv, callback):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("选择处理区域")
        self.dialog.geometry("1300x850")
        self.dialog.minsize(1000, 600)

        self.callback = callback
        self.image = image
        self.image_cv = image_cv
        self.selected_region = None

        self.setup_ui()
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.after(100, lambda: self.viewer.fit_to_window())

    def setup_ui(self):
        main = ttk.Frame(self.dialog)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctrl = ttk.Frame(main)
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        ttk.Label(ctrl, text="请在图片上拖动鼠标选择处理区域", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(ctrl, text="未选择区域", foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=20)

        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="确认", command=self.confirm, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)

        img_frame = ttk.LabelFrame(main, text="图片区域 (鼠标滚轮缩放，拖动选择区域)", padding=5)
        img_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = ttk.Frame(img_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(self.canvas_frame, bg='gray', highlightthickness=0)
        h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.image_canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.image_canvas.grid(row=0, column=0, sticky='nsew')
        h_scroll.grid(row=1, column=0, sticky='ew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.viewer = ImageViewer(self.image_canvas, self.on_region_selected, enable_drag=False)
        self.viewer.set_image(self.image)

    def on_region_selected(self, x1, y1, x2, y2):
        self.selected_region = (x1, y1, x2, y2)
        self.status_label.config(text=f"已选择: {x2 - x1}×{y2 - y1} 像素", foreground="green")

        self.image_canvas.delete("temp_region")
        cx1, cy1 = self.viewer.to_canvas(x1, y1)
        cx2, cy2 = self.viewer.to_canvas(x2, y2)
        self.image_canvas.create_rectangle(cx1, cy1, cx2, cy2, outline='blue', width=3, tags="temp_region")

    def confirm(self):
        if self.selected_region:
            x1, y1, x2, y2 = self.selected_region
            process_image = self.image.crop((x1, y1, x2, y2))
            process_cv = self.image_cv[y1:y2, x1:x2]
            self.callback(process_image, process_cv, (x1, y1))
            self.dialog.destroy()
        else:
            messagebox.showwarning("警告", "请先在图片上拖动选择区域！")


class SPADTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SPAD值预测工具")
        self.root.geometry("1600x1000")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.original_image = None
        self.original_cv = None
        self.process_image = None
        self.process_cv = None
        self.process_offset = (0, 0)
        self.predictor = SPADPredictor()
        self.active_model_type = 'linear'
        self.grid_regions = None
        self.grid_preview = None
        self.heatmap_image = None
        self.grid_info = None
        self.current_display_mode = "grid"
        self.show_gridlines = False
        self.gridline_ids = []
        self._rotation_lock = False

        self.use_green_filter = False
        self.green_threshold = 50

        self.font_cache = {}
        self.setup_ui()

    def on_closing(self):
        self.root.destroy()
        sys.exit(0)

    def get_font(self, size):
        if size in self.font_cache:
            return self.font_cache[size]
        system = platform.system()
        paths = {
            "Windows": "C:/Windows/Fonts/arial.ttf",
            "Darwin": "/System/Library/Fonts/Helvetica.ttc",
            "Linux": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        }
        try:
            font = ImageFont.truetype(paths.get(system, ""), size) if os.path.exists(
                paths.get(system, "")) else ImageFont.load_default()
        except (OSError, IOError):
            font = ImageFont.load_default()
        self.font_cache[size] = font
        return font

    def draw_gridlines(self):
        for gid in self.gridline_ids:
            self.image_canvas.delete(gid)
        self.gridline_ids.clear()

        if not self.show_gridlines:
            return

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        x1 = canvas_width // 3
        x2 = 2 * canvas_width // 3
        y1 = canvas_height // 3
        y2 = 2 * canvas_height // 3

        id1 = self.image_canvas.create_line(x1, 0, x1, canvas_height, fill='red', width=1, dash=(5, 5))
        id2 = self.image_canvas.create_line(x2, 0, x2, canvas_height, fill='red', width=1, dash=(5, 5))
        id3 = self.image_canvas.create_line(0, y1, canvas_width, y1, fill='red', width=1, dash=(5, 5))
        id4 = self.image_canvas.create_line(0, y2, canvas_width, y2, fill='red', width=1, dash=(5, 5))

        self.gridline_ids = [id1, id2, id3, id4]

        for gid in self.gridline_ids:
            self.image_canvas.tag_raise(gid)

    def toggle_gridlines(self):
        self.show_gridlines = not self.show_gridlines
        self.draw_gridlines()
        status = "开启" if self.show_gridlines else "关闭"
        self.status_var.set(f"参考线已{status}")

    def toggle_green_filter(self):
        self.use_green_filter = not self.use_green_filter
        status = "开启" if self.use_green_filter else "关闭"
        self.status_var.set(f"绿色背景过滤已{status}")

        if self.use_green_filter:
            self.status_var.set(f"绿色背景过滤已{status} (阈值: {self.green_threshold:.0f})")

        if self.predictor.calibration_points:
            messagebox.showinfo("提示", "背景过滤模式已更改，请重新进行校准")
            self.predictor = SPADPredictor()
            self.model_status_label.config(text="⚪ 未校准", foreground="red")
            self.update_fit_display()

    def set_custom_threshold(self):
        ThresholdDialog(self.root, self.green_threshold, self.update_threshold)

    def update_threshold(self, new_threshold):
        self.green_threshold = new_threshold
        self.status_var.set(f"绿色阈值已设置为 {self.green_threshold:.0f}")

        if self.predictor.calibration_points:
            messagebox.showinfo("提示", "绿色阈值已更改，请重新进行校准")
            self.predictor = SPADPredictor()
            self.model_status_label.config(text="⚪ 未校准", foreground="red")
            self.update_fit_display()

    def on_rotation_change(self, value):
        if self._rotation_lock:
            return
        self._rotation_lock = True

        angle = float(value)
        self.angle_value_label.config(text=f"{angle:.1f}°")
        self.viewer.set_rotation(angle)

        self.root.after(50, lambda: setattr(self, '_rotation_lock', False))

    def reset_rotation(self):
        if self._rotation_lock:
            return
        self._rotation_lock = True

        self.rotation_slider.set(0)
        self.angle_value_label.config(text="0.0°")
        self.viewer.set_rotation(0)

        self.grid_regions = None
        self.grid_preview = None
        self.heatmap_image = None
        self.preview_viewer.set_image(None)
        self.view_mode_label.config(text="当前: 无预览", foreground="gray")

        if self.original_image:
            self.process_image = self.original_image
            self.process_cv = self.original_cv
            self.viewer.set_image(self.original_image)

        self.root.after(50, lambda: setattr(self, '_rotation_lock', False))

    def on_model_changed(self):
        if not self.predictor.is_trained:
            return
        self.active_model_type = self.model_var.get()
        self.predictor.set_active_model(self.active_model_type)
        self.update_fit_display()
        self.update_scatter_plot()

    def open_elastic_settings(self):
        if not self.predictor.is_trained:
            messagebox.showinfo("提示", "请先进行校准！")
            return
        params = self.predictor.model_params['elastic']
        ElasticNetSettingsDialog(
            self.root,
            params['alpha'],
            params['l1_ratio'],
            self.on_elastic_params_changed
        )

    def on_elastic_params_changed(self, alpha, l1_ratio):
        self.predictor.set_elastic_params(alpha, l1_ratio)
        self.update_fit_display()
        self.update_scatter_plot()
        self.status_var.set(f"Elastic Net参数已更新 (α={alpha:.2f}, L1={l1_ratio:.2f})，模型已重新训练")

    def setup_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctrl = ttk.LabelFrame(main, text="控制面板", padding=10)
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(ctrl)
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="图片:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.input_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.input_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(row1, text="浏览", command=self.browse_image, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="加载", command=self.load_image, width=8).pack(side=tk.LEFT, padx=2)

        row_rotate = ttk.Frame(ctrl)
        row_rotate.pack(fill=tk.X, pady=5)
        ttk.Label(row_rotate, text="图片旋转:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(row_rotate, text="-180°", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.rotation_slider = ttk.Scale(row_rotate, from_=-180, to=180, orient=tk.HORIZONTAL, value=0, length=250,
                                         command=self.on_rotation_change)
        self.rotation_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Label(row_rotate, text="180°", font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
        self.angle_value_label = ttk.Label(row_rotate, text="0.0°", font=("Arial", 9, "bold"), width=8)
        self.angle_value_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(row_rotate, text="重置", command=self.reset_rotation, width=6).pack(side=tk.LEFT, padx=2)

        row_gridline = ttk.Frame(ctrl)
        row_gridline.pack(fill=tk.X, pady=5)
        self.gridline_var = tk.BooleanVar(value=False)
        self.gridline_check = ttk.Checkbutton(row_gridline, text="显示参考线", variable=self.gridline_var,
                                              command=self.toggle_gridlines)
        self.gridline_check.pack(side=tk.LEFT, padx=5)

        row_filter = ttk.Frame(ctrl)
        row_filter.pack(fill=tk.X, pady=5)
        self.filter_var = tk.BooleanVar(value=False)
        self.filter_check = ttk.Checkbutton(row_filter, text="绿色背景过滤", variable=self.filter_var,
                                            command=self.toggle_green_filter)
        self.filter_check.pack(side=tk.LEFT, padx=5)

        ttk.Label(row_filter, text=f"默认阈值: 50", font=("Arial", 9), foreground="gray").pack(side=tk.LEFT, padx=10)
        ttk.Button(row_filter, text="自定义阈值", command=self.set_custom_threshold, width=10).pack(side=tk.LEFT,
                                                                                                    padx=5)

        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(fill=tk.X, pady=10)

        self.btn_region = ttk.Button(btn_frame, text="1. 选择区域", command=self.select_region, width=16)
        self.btn_region.pack(side=tk.LEFT, padx=5)

        self.btn_calib = ttk.Button(btn_frame, text="2. 校准模式", command=self.open_calibration, width=16)
        self.btn_calib.pack(side=tk.LEFT, padx=5)

        self.btn_grid = ttk.Button(btn_frame, text="3. 划分网格", command=self.divide_grid, width=16)
        self.btn_grid.pack(side=tk.LEFT, padx=5)

        self.btn_predict = ttk.Button(btn_frame, text="4. 预测SPAD", command=self.predict_spad, width=16)
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_save = ttk.Button(btn_frame, text="5. 保存结果", command=self.save_results, width=16)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.model_status_label = ttk.Label(btn_frame, text="⚪ 未校准", foreground="red")
        self.model_status_label.pack(side=tk.LEFT, padx=20)

        info_frame = ttk.Frame(ctrl)
        info_frame.pack(fill=tk.X, pady=5)
        self.info_label = ttk.Label(info_frame, text="就绪", foreground="blue")
        self.info_label.pack(side=tk.LEFT)
        self.progress_label = ttk.Label(info_frame, text="", foreground="green")
        self.progress_label.pack(side=tk.RIGHT)

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(paned, text="处理图片区域 (鼠标滚轮缩放，拖拽移动图片，右键菜单)", padding=5)
        paned.add(left_frame, weight=3)

        self.canvas_frame = ttk.Frame(left_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(self.canvas_frame, bg='gray', highlightthickness=0)
        h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.image_canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.image_canvas.grid(row=0, column=0, sticky='nsew')
        h_scroll.grid(row=1, column=0, sticky='ew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)

        right_top = ttk.LabelFrame(right_frame, text="预览窗口", padding=5)
        right_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))

        switch_frame = ttk.Frame(right_top)
        switch_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.btn_grid_view = ttk.Button(switch_frame, text="网格预览", command=self.show_grid_preview, width=12)
        self.btn_grid_view.pack(side=tk.LEFT, padx=5)

        self.btn_heatmap_view = ttk.Button(switch_frame, text="热力图", command=self.show_heatmap, width=12)
        self.btn_heatmap_view.pack(side=tk.LEFT, padx=5)

        self.view_mode_label = ttk.Label(switch_frame, text="当前: 网格预览", foreground="blue")
        self.view_mode_label.pack(side=tk.LEFT, padx=10)

        self.preview_canvas_frame = ttk.Frame(right_top)
        self.preview_canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(self.preview_canvas_frame, bg='gray', highlightthickness=0)
        h2 = ttk.Scrollbar(self.preview_canvas_frame, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        v2 = ttk.Scrollbar(self.preview_canvas_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        self.preview_canvas.configure(xscrollcommand=h2.set, yscrollcommand=v2.set)
        self.preview_canvas.grid(row=0, column=0, sticky='nsew')
        h2.grid(row=1, column=0, sticky='ew')
        v2.grid(row=0, column=1, sticky='ns')
        self.preview_canvas_frame.grid_rowconfigure(0, weight=1)
        self.preview_canvas_frame.grid_columnconfigure(0, weight=1)

        model_frame = ttk.LabelFrame(right_frame, text="模型选择", padding=5)
        model_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 5))

        self.model_var = tk.StringVar(value='linear')
        ttk.Radiobutton(model_frame, text="线性回归 (Linear Regression)",
                        variable=self.model_var, value='linear',
                        command=self.on_model_changed).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(model_frame, text="Elastic Net",
                        variable=self.model_var, value='elastic',
                        command=self.on_model_changed).pack(anchor=tk.W, padx=5)

        elastic_btn_frame = ttk.Frame(model_frame)
        elastic_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(elastic_btn_frame, text="Elastic Net 参数设置",
                   command=self.open_elastic_settings, width=18).pack(side=tk.LEFT)

        right_bottom = ttk.LabelFrame(right_frame, text="拟合信息", padding=5)
        right_bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(right_bottom)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        info_tab = ttk.Frame(notebook)
        notebook.add(info_tab, text="方程信息")

        self.fit_text = tk.Text(info_tab, height=12, font=("Courier", 11))
        self.fit_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="标定点数据")

        cols = ('编号', 'R值', 'G值', 'B值', 'SPAD实测值')
        self.data_tree = ttk.Treeview(data_tab, columns=cols, show='headings', height=10)
        for c in cols:
            self.data_tree.heading(c, text=c)
            self.data_tree.column(c, width=100, anchor='center')

        data_scroll = ttk.Scrollbar(data_tab, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=data_scroll.set)
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        plot_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text="拟合图形")

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, plot_tab)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.viewer = ImageViewer(self.image_canvas, enable_drag=True)
        self.preview_viewer = ImageViewer(self.preview_canvas, enable_drag=False)

        self.image_canvas.bind("<Configure>", lambda e: self.draw_gridlines())

        bottom = ttk.Frame(main)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        self.status_var = tk.StringVar(
            value="就绪 | 操作流程: 1.选择区域 2.校准模式添加标定点 3.划分网格 4.预测SPAD 5.保存结果")
        ttk.Label(bottom, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)

    def show_grid_preview(self):
        if self.grid_preview:
            self.preview_viewer.set_image(self.grid_preview)
            self.current_display_mode = "grid"
            self.view_mode_label.config(text="当前: 网格预览", foreground="blue")
            self.status_var.set("切换到网格预览模式")

    def show_heatmap(self):
        if self.heatmap_image:
            self.preview_viewer.set_image(self.heatmap_image)
            self.current_display_mode = "heatmap"
            self.view_mode_label.config(text="当前: 热力图", foreground="blue")
            self.status_var.set("切换到热力图模式")
        else:
            messagebox.showinfo("提示", "请先预测SPAD值生成热力图！")

    def browse_image(self):
        p = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if p:
            self.input_var.set(p)

    def load_image(self):
        p = self.input_var.get()
        if not p or not os.path.exists(p):
            messagebox.showwarning("警告", "请选择有效的图片文件！")
            return
        try:
            self.original_image = Image.open(p)
            self.original_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            w, h = self.original_image.size
            self.info_label.config(text=f"已加载: {os.path.basename(p)} | {w}×{h}")
            self.status_var.set(f"图片已加载: {w}×{h}，请点击【选择区域】")

            self.process_image = self.original_image
            self.process_cv = self.original_cv
            self.process_offset = (0, 0)
            self.predictor = SPADPredictor()
            self.active_model_type = 'linear'
            self.grid_regions = None
            self.grid_preview = None
            self.heatmap_image = None
            self.grid_info = None
            self.viewer.set_image(self.original_image)
            self.preview_viewer.set_image(None)
            self.update_fit_display()
            self.model_status_label.config(text="⚪ 未校准", foreground="red")
            self.view_mode_label.config(text="当前: 无预览", foreground="gray")
            self.rotation_slider.set(0)
            self.angle_value_label.config(text="0.0°")
            self.draw_gridlines()
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def select_region(self):
        if not self.original_image:
            messagebox.showwarning("警告", "请先加载图片！")
            return
        rotated_image = self.viewer.get_rotated_image()
        rotated_cv = cv2.cvtColor(np.array(rotated_image), cv2.COLOR_RGB2BGR)
        RegionSelectDialog(self.root, rotated_image, rotated_cv, self.on_region_selected)

    def on_region_selected(self, process_image, process_cv, offset):
        self.process_image = process_image
        self.process_cv = process_cv
        self.process_offset = offset
        self.viewer.set_image(process_image)
        self.predictor = SPADPredictor()
        self.active_model_type = 'linear'
        self.grid_regions = None
        self.grid_preview = None
        self.heatmap_image = None
        self.grid_info = None
        self.preview_viewer.set_image(None)
        self.update_fit_display()
        self.model_status_label.config(text="⚪ 未校准", foreground="red")
        self.status_var.set(f"已选择处理区域: {process_image.width}×{process_image.height} 像素")
        self.info_label.config(text=f"处理区域: {process_image.width}×{process_image.height}")
        self.view_mode_label.config(text="当前: 无预览", foreground="gray")
        self.draw_gridlines()

    def open_calibration(self):
        if not self.process_image:
            messagebox.showwarning("警告", "请先选择处理区域！")
            return
        CalibrationDialog(self.root, self.process_image, self.process_cv, self.on_calibration_done,
                          self.use_green_filter, self.green_threshold)

    def on_calibration_done(self, predictor):
        self.predictor = predictor
        self.active_model_type = 'linear'
        self.model_var.set('linear')

        self.update_fit_display()
        self.model_status_label.config(text="✅ 已校准", foreground="green")

        filter_status = " (已启用绿色过滤)" if self.use_green_filter else ""
        self.status_var.set(f"校准完成！已使用 {len(predictor.calibration_points)} 个标定点{filter_status}")

        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        for p in predictor.calibration_points:
            self.data_tree.insert('', 'end',
                                  values=(p['index'], f"{p['R']:.2f}", f"{p['G']:.2f}", f"{p['B']:.2f}", p['SPAD']))

        self.update_scatter_plot()

        linear_metrics = predictor.get_metrics('linear')
        elastic_metrics = predictor.get_metrics('elastic')
        msg = (f"模型训练完成！\n\n"
               f"线性回归 R² = {linear_metrics['r2']:.4f}, MAE = {linear_metrics['mae']:.4f}\n"
               f"Elastic Net R² = {elastic_metrics['r2']:.4f}, MAE = {elastic_metrics['mae']:.4f}\n\n"
               f"当前使用: 线性回归\n可在右侧面板切换模型")
        if self.use_green_filter:
            msg += f"\n\n绿色过滤阈值: {self.green_threshold:.0f}"
        messagebox.showinfo("校准完成", msg)

    def update_scatter_plot(self):
        if not self.predictor.is_trained:
            return

        y_true = [p['SPAD'] for p in self.predictor.calibration_points]

        linear_model = self.predictor.models['linear']
        elastic_model = self.predictor.models['elastic']

        X = pd.DataFrame(self.predictor.calibration_points)[['R', 'G', 'B']]
        y_pred_linear = linear_model.predict(X)
        y_pred_elastic = elastic_model.predict(X)

        self.ax.clear()

        self.ax.scatter(y_true, y_pred_linear, alpha=0.7, c='blue', s=60,
                        marker='o', label=f'Linear (R²={self.predictor.model_metrics["linear"]["r2"]:.3f})')
        self.ax.scatter(y_true, y_pred_elastic, alpha=0.7, c='red', s=60,
                        marker='^', label=f'Elastic Net (R²={self.predictor.model_metrics["elastic"]["r2"]:.3f})')

        min_val = min(min(y_true), min(y_pred_linear), min(y_pred_elastic)) - 5
        max_val = max(max(y_true), max(y_pred_linear), max(y_pred_elastic)) + 5
        self.ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='理想线')

        self.ax.set_xlabel("实测SPAD值", fontsize=11)
        self.ax.set_ylabel("预测SPAD值", fontsize=11)
        self.ax.set_title("SPAD预测模型对比", fontsize=12)
        self.ax.legend(fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.canvas_plot.draw()

    def update_fit_display(self):
        if self.predictor.is_trained:
            linear_metrics = self.predictor.get_metrics('linear')
            elastic_metrics = self.predictor.get_metrics('elastic')

            info = "=" * 60 + "\n"
            info += "SPAD预测模型对比\n"
            info += "=" * 60 + "\n\n"

            active_marker = "▶ " if self.active_model_type == 'linear' else "  "
            info += f"{active_marker}线性回归 (Linear Regression)\n"
            info += f"  {self.predictor.get_equation_string('linear')}\n"
            info += f"  R² = {linear_metrics['r2']:.6f}  |  MAE = {linear_metrics['mae']:.6f}\n\n"

            active_marker = "▶ " if self.active_model_type == 'elastic' else "  "
            params = self.predictor.model_params['elastic']
            info += f"{active_marker}Elastic Net (α={params['alpha']:.2f}, L1={params['l1_ratio']:.2f})\n"
            info += f"  {self.predictor.get_equation_string('elastic')}\n"
            info += f"  R² = {elastic_metrics['r2']:.6f}  |  MAE = {elastic_metrics['mae']:.6f}\n\n"

            info += f"标定点数量: {len(self.predictor.calibration_points)}\n"
            if self.use_green_filter:
                info += f"绿色过滤阈值: {self.green_threshold:.0f}\n"
            info += "\n" + "=" * 60 + "\n"
            info += "▶ 表示当前选中的模型，用于SPAD预测"

            self.fit_text.delete(1.0, tk.END)
            self.fit_text.insert(tk.END, info)
        else:
            self.fit_text.delete(1.0, tk.END)
            self.fit_text.insert(tk.END,
                                 "请先进行校准...\n\n点击【校准模式】按钮，在图片上选择标定点并输入SPAD值\n至少需要3个标定点才能拟合方程")

    def divide_grid(self):
        if not self.process_image:
            messagebox.showwarning("警告", "请先选择处理区域！")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("网格设置")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (350 // 2)
        y = (dialog.winfo_screenheight() // 2) - (200 // 2)
        dialog.geometry(f"350x200+{x}+{y}")

        ttk.Label(dialog, text="请输入网格行列数:", font=("Arial", 11)).pack(pady=15)

        frame = ttk.Frame(dialog)
        frame.pack(pady=10)
        ttk.Label(frame, text="行数:").pack(side=tk.LEFT, padx=5)
        rows_var = tk.StringVar(value="4")
        rows_spin = ttk.Spinbox(frame, from_=1, to=50, textvariable=rows_var, width=8)
        rows_spin.pack(side=tk.LEFT, padx=5)

        frame2 = ttk.Frame(dialog)
        frame2.pack(pady=10)
        ttk.Label(frame2, text="列数:").pack(side=tk.LEFT, padx=5)
        cols_var = tk.StringVar(value="5")
        cols_spin = ttk.Spinbox(frame2, from_=1, to=50, textvariable=cols_var, width=8)
        cols_spin.pack(side=tk.LEFT, padx=5)

        def confirm():
            rows = int(rows_var.get())
            cols = int(cols_var.get())
            self.do_divide_grid(rows, cols)
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="确定", command=confirm, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=10)

    def do_divide_grid(self, rows, cols):
        w, h = self.process_image.size
        rw, rh = w / cols, h / rows
        result = self.process_image.copy()
        draw = ImageDraw.Draw(result)

        for i in range(1, cols):
            draw.line([(i * rw, 0), (i * rw, h)], fill=(255, 0, 0), width=2)
        for i in range(1, rows):
            draw.line([(0, i * rh), (w, i * rh)], fill=(255, 0, 0), width=2)

        font = self.get_font(max(12, int(min(rw, rh) * 0.35)))
        num = 1
        for row in range(rows):
            for col in range(cols):
                cx, cy = col * rw + rw / 2, row * rh + rh / 2
                text = str(num)
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    draw.text((cx - text_w / 2, cy - text_h / 2), text, fill=(255, 0, 0), font=font)
                except Exception:
                    draw.text((cx - 8, cy - 8), text, fill=(255, 0, 0), font=font)
                num += 1

        draw.text((10, 10), f"网格: {rows}×{cols} | 每块: {rw:.1f}×{rh:.1f} px", fill=(255, 255, 0))

        self.grid_regions = []
        num = 1
        for row in range(rows):
            for col in range(cols):
                self.grid_regions.append({
                    'id': num, 'row': row + 1, 'col': col + 1,
                    'x1': int(col * rw), 'y1': int(row * rh),
                    'x2': int((col + 1) * rw), 'y2': int((row + 1) * rh),
                    'spad': None, 'R': None, 'G': None, 'B': None
                })
                num += 1

        self.grid_info = {
            'rows': rows,
            'cols': cols,
            'rw': rw,
            'rh': rh,
            'w': w,
            'h': h
        }

        self.grid_preview = result
        self.preview_viewer.set_image(result)
        self.current_display_mode = "grid"
        self.view_mode_label.config(text="当前: 网格预览", foreground="blue")
        self.status_var.set(f"网格划分完成: {rows}行×{cols}列 = {rows * cols}个矩形")

        if self.predictor.is_trained:
            self.predict_spad()

    def predict_spad(self):
        if not self.predictor.is_trained:
            messagebox.showwarning("警告", "请先进行校准！点击【校准模式】添加至少3个标定点")
            return
        if not self.grid_regions:
            messagebox.showwarning("警告", "请先划分网格！点击【划分网格】")
            return

        spad_values = []
        for r in self.grid_regions:
            roi = self.process_cv[r['y1']:r['y2'], r['x1']:r['x2']]
            r_val, g_val, b_val = calculate_rgb_with_filter(roi, self.use_green_filter, self.green_threshold)
            r['R'] = r_val
            r['G'] = g_val
            r['B'] = b_val
            spad = self.predictor.predict(r_val, g_val, b_val)
            r['spad'] = spad
            spad_values.append(spad)

        self.generate_heatmap(spad_values)

        model_name = "线性回归" if self.active_model_type == 'linear' else "Elastic Net"
        filter_status = " (已启用绿色过滤)" if self.use_green_filter else ""
        self.status_var.set(f"已完成 {len(self.grid_regions)} 个区域的SPAD预测 [{model_name}]{filter_status}")
        messagebox.showinfo("预测完成",
                            f"已成功预测 {len(self.grid_regions)} 个区域的SPAD值！\n"
                            f"当前模型: {model_name}\n\n"
                            f"请点击【保存结果】导出数据")

    def add_legend_to_heatmap(self, heatmap):
        img_width = heatmap.width
        img_height = heatmap.height

        legend_area_width = 80
        new_width = img_width + legend_area_width
        new_image = Image.new('RGB', (new_width, img_height), 'white')
        new_image.paste(heatmap, (0, 0))

        draw = ImageDraw.Draw(new_image)

        legend_x = img_width + 10
        legend_y = 10
        legend_width = 60
        legend_height = 200

        draw.rectangle([legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                       fill=(245, 245, 245), outline=(100, 100, 100), width=1)

        colors = [
            (0, 100, 0),
            (0, 128, 0),
            (255, 255, 0),
            (255, 165, 0),
            (255, 0, 0)
        ]

        bar_width = 20
        bar_height = 25
        bar_x = legend_x + (legend_width - bar_width) // 2
        bar_start_y = legend_y + 30

        for i, color in enumerate(colors):
            y_pos = bar_start_y + i * bar_height
            draw.rectangle([bar_x, y_pos, bar_x + bar_width, y_pos + bar_height],
                           fill=color, outline=(0, 0, 0), width=1)

        font_small = self.get_font(10)
        arrow_x = legend_x + legend_width // 2
        draw.text((arrow_x - 8, bar_start_y - 18), "High", fill=(0, 0, 0), font=font_small)
        draw.line([(arrow_x, bar_start_y - 12), (arrow_x, bar_start_y - 3)], fill=(0, 0, 0), width=1)
        draw.polygon([(arrow_x, bar_start_y - 1), (arrow_x - 3, bar_start_y - 5), (arrow_x + 3, bar_start_y - 5)],
                     fill=(0, 0, 0))

        draw.text((arrow_x - 6, bar_start_y + 5 * bar_height + 5), "Low", fill=(0, 0, 0), font=font_small)
        draw.line([(arrow_x, bar_start_y + 5 * bar_height + 10), (arrow_x, bar_start_y + 5 * bar_height + 18)],
                  fill=(0, 0, 0), width=1)
        draw.polygon([(arrow_x, bar_start_y + 5 * bar_height + 20), (arrow_x - 3, bar_start_y + 5 * bar_height + 16),
                      (arrow_x + 3, bar_start_y + 5 * bar_height + 16)], fill=(0, 0, 0))

        return new_image

    def generate_heatmap(self, spad_values):
        if not self.grid_regions or not self.grid_info:
            return

        rows = self.grid_info['rows']
        cols = self.grid_info['cols']
        rw = self.grid_info['rw']
        rh = self.grid_info['rh']

        heatmap = Image.new('RGB', (int(cols * rw), int(rows * rh)), 'white')
        draw = ImageDraw.Draw(heatmap)

        spad_list = [r['spad'] for r in self.grid_regions]
        min_spad = min(spad_list)
        max_spad = max(spad_list)
        range_spad = max_spad - min_spad if max_spad != min_spad else 1

        colors = [
            (255, 0, 0),
            (255, 165, 0),
            (255, 255, 0),
            (0, 128, 0),
            (0, 100, 0)
        ]

        for region in self.grid_regions:
            spad = region['spad']
            norm = (spad - min_spad) / range_spad

            if norm <= 0.25:
                t = norm / 0.25
                color = self.interpolate_color(colors[0], colors[1], t)
            elif norm <= 0.5:
                t = (norm - 0.25) / 0.25
                color = self.interpolate_color(colors[1], colors[2], t)
            elif norm <= 0.75:
                t = (norm - 0.5) / 0.25
                color = self.interpolate_color(colors[2], colors[3], t)
            else:
                t = (norm - 0.75) / 0.25
                color = self.interpolate_color(colors[3], colors[4], t)

            x1, y1 = region['x1'], region['y1']
            x2, y2 = region['x2'], region['y2']
            draw.rectangle([x1, y1, x2, y2], fill=color)

        for i in range(1, cols):
            x = i * rw
            draw.line([(x, 0), (x, rows * rh)], fill=(0, 0, 0), width=1)
        for i in range(1, rows):
            y = i * rh
            draw.line([(0, y), (cols * rw, y)], fill=(0, 0, 0), width=1)

        font_size = max(10, int(min(rw, rh) * 0.25))
        font = self.get_font(font_size)

        for region in self.grid_regions:
            cx = (region['x1'] + region['x2']) / 2
            cy = (region['y1'] + region['y2']) / 2
            text = f"{region['spad']:.1f}"

            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.text((cx - text_w / 2, cy - text_h / 2), text, fill=(0, 0, 0), font=font)
            except Exception:
                draw.text((cx - 15, cy - 8), text, fill=(0, 0, 0), font=font)

        heatmap_with_legend = self.add_legend_to_heatmap(heatmap)
        self.heatmap_image = heatmap_with_legend

        if self.current_display_mode == "heatmap":
            self.preview_viewer.set_image(heatmap_with_legend)

        self.status_var.set(f"热力图已生成，SPAD范围: {min_spad:.1f} - {max_spad:.1f}")

    def interpolate_color(self, color1, color2, t):
        return tuple(int(color1[i] + (color2[i] - color1[i]) * t) for i in range(3))

    def save_results(self):
        if not self.grid_regions or not any(r['spad'] for r in self.grid_regions):
            messagebox.showwarning("警告", "请先预测SPAD值！点击【预测SPAD】")
            return

        save_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
        )

        if not save_path:
            return

        try:
            test_file = open(save_path, 'wb')
            test_file.close()
        except PermissionError:
            messagebox.showerror("错误", f"文件已被其他程序打开！\n\n请关闭Excel或其他程序后重试。\n\n{save_path}")
            return

        df = pd.DataFrame(self.grid_regions)
        df = df[['id', 'R', 'G', 'B', 'spad']]
        df.columns = ['编号', 'R平均值', 'G平均值', 'B平均值', 'SPAD预测值']

        try:
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='SPAD预测结果', index=False)

                if self.predictor.is_trained:
                    linear_metrics = self.predictor.get_metrics('linear')
                    elastic_metrics = self.predictor.get_metrics('elastic')
                    elastic_params = self.predictor.model_params['elastic']

                    model_info = {
                        '模型': ['线性回归', 'Elastic Net'],
                        'R系数(a)': [linear_metrics['coef'][0], elastic_metrics['coef'][0]],
                        'G系数(b)': [linear_metrics['coef'][1], elastic_metrics['coef'][1]],
                        'B系数(c)': [linear_metrics['coef'][2], elastic_metrics['coef'][2]],
                        '截距(d)': [linear_metrics['intercept'], elastic_metrics['intercept']],
                        'R²': [linear_metrics['r2'], elastic_metrics['r2']],
                        'MAE': [linear_metrics['mae'], elastic_metrics['mae']],
                        '当前使用': ['是' if self.active_model_type == 'linear' else '否',
                                    '是' if self.active_model_type == 'elastic' else '否']
                    }

                    model_df = pd.DataFrame(model_info)
                    model_df.to_excel(writer, sheet_name='模型对比', index=False)

                    if self.use_green_filter:
                        filter_df = pd.DataFrame({
                            '参数': ['绿色过滤', '阈值'],
                            '数值': ['是', self.green_threshold]
                        })
                        filter_df.to_excel(writer, sheet_name='过滤设置', index=False)

                    elastic_params_df = pd.DataFrame({
                        '参数': ['Alpha', 'L1 Ratio'],
                        '数值': [elastic_params['alpha'], elastic_params['l1_ratio']]
                    })
                    elastic_params_df.to_excel(writer, sheet_name='Elastic Net参数', index=False)

            saved_files = []

            if self.grid_preview:
                img_path = save_path.replace('.xlsx', '_网格预览.png')
                self.grid_preview.save(img_path)
                saved_files.append(img_path)

            if self.heatmap_image:
                heatmap_path = save_path.replace('.xlsx', '_热力图.png')
                self.heatmap_image.save(heatmap_path)
                saved_files.append(heatmap_path)

            msg = f"结果已保存至:\n{save_path}\n\n"
            if saved_files:
                msg += "图片文件:\n" + "\n".join(saved_files)
            messagebox.showinfo("保存成功", msg)
        except PermissionError:
            messagebox.showerror("错误", f"无法保存文件！\n\n请确保文件未被其他程序打开，然后重试。\n\n{save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SPADTool()
    app.run()
