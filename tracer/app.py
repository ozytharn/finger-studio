from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, messagebox, ttk
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from .camera import CameraProcessor
from .core import FINGER_LANDMARKS, FINGER_NAMES, HandObservation, StableTracker, Trail, map_point


class TracerApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Finger Studio")
        self.root.minsize(980, 650)
        self.processor: Optional[CameraProcessor] = None
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.frames: "queue.Queue[Tuple[np.ndarray, list[HandObservation]]]" = queue.Queue(maxsize=1)
        self.tracker = StableTracker()
        self.trails: Dict[str, Trail] = {finger: Trail() for finger in FINGER_NAMES}
        self.last_points: Dict[str, Tuple[float, float]] = {}
        self.last_frame_time = time.monotonic()
        self.fps = 0.0
        self.bounds = (0.0, 0.0, 1.0, 1.0)
        self.trail_color = "#00ff66"
        self.inference_ms = 0.0
        self.trace_state = "stopped"
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self._poll_frames)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        palette = {"bg": "#0C1020", "sidebar": "#12182A", "panel": "#182239",
                   "control": "#202C48", "border": "#2C3A5B", "text": "#EEF3FF",
                   "secondary": "#A8B6D7", "muted": "#7180A4", "accent": "#71E7F4",
                   "success": "#64E6B0", "warning": "#F2C477", "error": "#FF8798"}
        self.palette = palette
        style.theme_use("clam")
        style.configure(".", background=palette["sidebar"], foreground=palette["text"],
                        font=("Segoe UI", 10))
        style.configure("TFrame", background=palette["sidebar"])
        style.configure("Workspace.TFrame", background="#090D19")
        style.configure("Title.TLabel", background=palette["sidebar"], foreground=palette["text"],
                        font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=palette["sidebar"], foreground=palette["secondary"])
        style.configure("Section.TLabelframe", background=palette["panel"], foreground=palette["text"],
                        bordercolor=palette["border"], padding=14)
        style.configure("Section.TLabelframe.Label", background=palette["panel"],
                        foreground=palette["secondary"],                         font=("Segoe UI", 9, "bold"))
        style.configure("TLabel", background=palette["sidebar"], foreground=palette["text"])
        style.configure("TButton", background=palette["control"], foreground=palette["text"],
                        bordercolor=palette["border"], padding=(11, 8), borderwidth=1)
        style.map("TButton", background=[("active", "#2C3A5B")],
                  foreground=[("active", palette["text"])])
        style.configure("Primary.TButton", background=palette["accent"], foreground="#07131F",
                        font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.map("Primary.TButton", background=[("active", "#A0F4FA")])
        style.configure("TCheckbutton", background=palette["panel"], foreground=palette["text"])
        style.configure("TCombobox", fieldbackground=palette["control"],
                        background=palette["control"], foreground=palette["text"])
        self.root.configure(background=palette["bg"])
        self.root.geometry("1360x860")
        self.root.minsize(1050, 680)

        top = tk.Frame(self.root, bg=palette["sidebar"], height=78,
                       highlightbackground=palette["border"], highlightthickness=1)
        top.pack(fill=tk.X)
        tk.Label(top, text="FS / 01", bg=palette["sidebar"], fg=palette["accent"],
                 font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=(20, 18))
        title = tk.Frame(top, bg=palette["sidebar"])
        title.pack(side=tk.LEFT, pady=8)
        tk.Label(title, text="Finger Studio", bg=palette["sidebar"], fg=palette["text"],
                 font=("Segoe UI", 19, "bold")).pack(anchor=tk.W)
        tk.Label(title, text="Precision fingertip tracing workspace", bg=palette["sidebar"],
                 fg=palette["secondary"], font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.connection_var = tk.StringVar(value="●  Camera connecting")
        tk.Label(top, textvariable=self.connection_var, bg=palette["sidebar"],
                 fg=palette["success"], font=("Segoe UI", 10, "bold")).pack(side=tk.RIGHT, padx=24)

        split = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        split.pack(fill=tk.BOTH, expand=True, padx=24, pady=(20, 24))
        camera_card = ttk.Frame(split, style="Workspace.TFrame", padding=12)
        panel_outer = ttk.Frame(split, padding=(20, 0, 0, 0))
        split.add(camera_card, weight=4)
        split.add(panel_outer, weight=1)
        self.preview = tk.Label(camera_card, background="#090D19", cursor="crosshair",
                                highlightbackground=palette["border"], highlightthickness=1)
        self.preview.pack(fill=tk.BOTH, expand=True)
        self.preview.bind("<Configure>", lambda _event: setattr(self, "_preview_size", True))

        panel = tk.Frame(panel_outer, bg=palette["sidebar"])
        panel.pack(fill=tk.BOTH, expand=True)
        header = tk.Frame(panel, bg=palette["sidebar"])
        header.pack(fill=tk.X, pady=(0, 14))
        tk.Label(header, text="Select your hand", bg=palette["sidebar"], fg=palette["text"],
                 font=("Segoe UI", 17, "bold")).pack(anchor=tk.W)
        tk.Label(header, text="Choose a fingertip to trace", bg=palette["sidebar"],
                 fg=palette["secondary"], font=("Segoe UI", 9)).pack(anchor=tk.W)

        self.hand_var = tk.StringVar(value="Either")
        self.finger_vars = {finger: tk.BooleanVar(value=finger == "Index") for finger in FINGER_NAMES}
        self.status_var = tk.StringVar(value="Starting camera...")
        self.position_var = tk.StringVar(value="Position: -")
        self.fps_var = tk.StringVar(value="FPS: -")
        self.resolution_var = tk.StringVar(value="Camera: starting...")
        self.landmarks_var = tk.BooleanVar(value=True)
        self.mirror_var = tk.BooleanVar(value=True)

        hand_card = ttk.LabelFrame(panel, text="FINGER FIELD  •  SELECT A TRACE POINT",
                                   style="Section.TLabelframe")
        hand_card.pack(fill=tk.X, pady=(0, 8))
        self.hand_diagram = tk.Canvas(hand_card, width=520, height=330, background=palette["panel"],
                                      highlightthickness=0, takefocus=True)
        self.hand_diagram.pack(fill=tk.X)
        self.hand_diagram.bind("<Key-1>", lambda _event: self._keyboard_select("Index"))
        self.hand_diagram.bind("<Key-2>", lambda _event: self._keyboard_select("Middle"))
        self.hand_diagram.bind("<Key-3>", lambda _event: self._keyboard_select("Ring"))
        self.hand_diagram.bind("<Key-4>", lambda _event: self._keyboard_select("Pinky"))
        self.hand_diagram.bind("<Key-5>", lambda _event: self._keyboard_select("Thumb"))
        self.hand_diagram.bind("<FocusIn>", lambda _event: self.status_var.set(
            "Finger selector focused • press 1-5 to choose Index, Middle, Ring, Pinky, Thumb"))
        self._draw_hand_diagram()
        self.selection_summary = ttk.Label(hand_card, text="Selected: Index", style="Subtitle.TLabel")
        self.selection_summary.pack(anchor=tk.W, pady=(4, 0))

        hand_section = ttk.LabelFrame(panel, text="HAND SOURCE  •  ACTIVE TRACKING SIDE", style="Section.TLabelframe")
        hand_section.pack(fill=tk.X, pady=(0, 8))
        self._combo(hand_section, "Active hand", self.hand_var, ("Either", "Left", "Right"), self._selection_changed)

        controls = ttk.LabelFrame(panel, text="TRACE CONTROL  •  LIVE STATE", style="Section.TLabelframe")
        controls.pack(fill=tk.X, pady=(0, 8))
        buttons = ttk.Frame(controls)
        buttons.pack(fill=tk.X)
        self.primary_button = ttk.Button(buttons, text="Start tracing",
                                         style="Primary.TButton", command=self._primary_trace_action)
        self.primary_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(buttons, text="Stop", command=self._stop_trace).pack(side=tk.LEFT)
        ttk.Button(controls, text="Clear all trails", command=self._clear_trail).pack(fill=tk.X, pady=(8, 0))

        options = ttk.LabelFrame(panel, text="TRAIL APPEARANCE  •  MATERIAL", style="Section.TLabelframe")
        options.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(options, text="Trail thickness").pack(anchor=tk.W)
        self.thickness = tk.IntVar(value=5)
        ttk.Scale(options, from_=1, to=25, variable=self.thickness, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Button(options, text="Choose trail color", command=self._choose_color).pack(fill=tk.X, pady=5)
        ttk.Checkbutton(options, text="Show landmarks", variable=self.landmarks_var).pack(anchor=tk.W)
        ttk.Checkbutton(options, text="Mirror preview", variable=self.mirror_var).pack(anchor=tk.W)
        area = ttk.LabelFrame(panel, text="ADVANCED SETTINGS", style="Section.TLabelframe")
        area.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(area, text="Adjust only if you want a smaller active region", style="Subtitle.TLabel").pack(anchor=tk.W)
        self.bound_vars = [tk.DoubleVar(value=value) for value in self.bounds]
        for label, variable in zip(("Left", "Top", "Right", "Bottom"), self.bound_vars):
            row = ttk.Frame(area)
            row.pack(fill=tk.X)
            ttk.Label(row, text=label, width=8).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=variable, width=8).pack(side=tk.RIGHT)
        ttk.Button(area, text="Use full camera frame", command=self._full_frame).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(area, text="Apply camera range", command=self._apply_bounds).pack(fill=tk.X, pady=(4, 0))
        status = tk.Frame(panel, bg=palette["panel"], highlightbackground=palette["border"],
                          highlightthickness=1, padx=12, pady=10)
        status.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(status, textvariable=self.status_var, wraplength=260).pack(anchor=tk.W)
        ttk.Label(panel, textvariable=self.position_var).pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(panel, textvariable=self.fps_var).pack(anchor=tk.W)
        ttk.Label(panel, textvariable=self.resolution_var, style="Subtitle.TLabel").pack(anchor=tk.W)
        self._selection_changed()
        self.root.bind("<space>", lambda _event: self._primary_trace_action())
        self.root.bind("<Control-l>", lambda _event: self._clear_trail())

    def _draw_hand_diagram(self) -> None:
        canvas = self.hand_diagram
        canvas.delete("all")
        asset = Path(__file__).parent.parent / "assets" / "hand_selector_new_hologram.png"
        source = Image.open(asset).convert("RGBA")
        self._hand_selector_images = []
        self._draw_photo_selector(canvas, source)

    def _draw_hand_asset(self, canvas: tk.Canvas, image: Image.Image, ox: int, oy: int,
                         hand: str, mirrored: bool) -> None:
        selected_hand = self.hand_var.get() == hand
        photo = ImageTk.PhotoImage(image)
        self._hand_selector_images.append(photo)
        canvas.create_image(ox, oy, image=photo, anchor=tk.NW)
        # These polygons are authored in the SVG's 220x280 coordinate space and
        # transformed by the exact scale used for the displayed raster asset.
        regions = {
            "Thumb": [(23, 145), (39, 143), (57, 157), (73, 176), (77, 198),
                      (68, 214), (51, 208), (39, 193), (25, 178)],
            "Index": [(53, 77), (53, 67), (59, 60), (68, 59), (78, 66),
                      (82, 76), (84, 136), (76, 160), (63, 159), (56, 145)],
            "Middle": [(84, 43), (84, 32), (91, 25), (100, 24), (109, 31),
                       (115, 42), (116, 136), (108, 169), (96, 168), (87, 150)],
            "Ring": [(116, 54), (118, 43), (125, 36), (134, 35), (143, 42),
                     (149, 53), (150, 141), (143, 174), (130, 173), (120, 153)],
            "Pinky": [(150, 73), (154, 63), (160, 57), (169, 57), (178, 64),
                      (183, 75), (178, 155), (171, 184), (159, 182), (152, 158)],
        }
        scale_x, scale_y = 168 / 220, 214 / 280
        for finger, points in regions.items():
            if mirrored:
                points = [(220 - x, y) for x, y in points]
            points = [(ox + x * scale_x, oy + y * scale_y) for x, y in points]
            selected = self.finger_vars[finger].get() and selected_hand
            item = canvas.create_polygon(
                points, fill=self.palette["accent"] if selected else "",
                stipple="gray50" if selected else "",
                outline=self.palette["accent"] if selected else "",
                width=2, tags=("hand-finger", hand, finger))
            canvas.tag_bind(item, "<Button-1>",
                            lambda _event, h=hand, f=finger: self._toggle_hand_finger(h, f))
            canvas.tag_bind(item, "<Enter>",
                            lambda _event, i=item: self._finger_hover(i, True))
            canvas.tag_bind(item, "<Leave>",
                            lambda _event, i=item, is_selected=selected:
                            self._finger_hover(i, False, is_selected))
        canvas.create_text(ox + 84, oy + 232, text=hand,
                           fill=self.palette["secondary"], font=("Segoe UI", 8))

    def _draw_photo_selector(self, canvas: tk.Canvas, source: Image.Image) -> None:
        source_w, source_h = source.size
        image_x, image_y = 2, 30
        scale = min(516 / source_w, 292 / source_h)
        display_size = (round(source_w * scale), round(source_h * scale))
        image = source.resize(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._hand_selector_images.append(photo)
        canvas.create_image(image_x, image_y, image=photo, anchor=tk.NW)
        canvas.create_text(112, 18, text="LEFT HAND", fill=self.palette["secondary"],
                           font=("Segoe UI", 8, "bold"))
        canvas.create_text(407, 18, text="RIGHT HAND", fill=self.palette["secondary"],
                           font=("Segoe UI", 8, "bold"))

        # Polygons follow the new supplied image's native 738x414 coordinate
        # space and use the exact uniform display transform above.
        regions = {
            "Left": {
                "Pinky": [(62, 96), (72, 85), (91, 87), (104, 105),
                          (111, 202), (101, 229), (84, 232), (71, 214)],
                "Ring": [(99, 32), (108, 22), (125, 24), (137, 43),
                         (140, 200), (130, 224), (113, 225), (102, 205)],
                "Middle": [(140, 13), (150, 5), (166, 7), (177, 26),
                           (178, 198), (168, 222), (151, 222), (143, 202)],
                "Index": [(188, 25), (198, 16), (214, 19), (225, 38),
                          (225, 201), (215, 224), (199, 224), (190, 204)],
                "Thumb": [(230, 176), (244, 168), (260, 176), (291, 205),
                          (307, 224), (294, 252), (275, 274), (253, 292),
                          (235, 279), (245, 247)],
            },
            "Right": {
                "Index": [(488, 39), (498, 29), (514, 31), (526, 49),
                          (528, 204), (518, 227), (502, 226), (491, 207)],
                "Middle": [(538, 19), (548, 10), (564, 12), (576, 31),
                           (577, 203), (567, 226), (551, 225), (541, 205)],
                "Ring": [(585, 31), (595, 21), (611, 24), (623, 43),
                         (623, 207), (613, 229), (597, 228), (588, 208)],
                "Pinky": [(632, 97), (642, 86), (657, 89), (667, 108),
                          (668, 208), (658, 231), (643, 230), (634, 211)],
                "Thumb": [(410, 204), (425, 194), (441, 202), (470, 231),
                          (492, 257), (486, 285), (468, 306), (448, 325),
                          (430, 312), (440, 279)],
            },
        }
        for hand, fingers in regions.items():
            selected_hand = self.hand_var.get() == hand
            for finger, source_points in fingers.items():
                points = [(image_x + x * scale, image_y + y * scale)
                          for x, y in source_points]
                selected = self.finger_vars[finger].get() and selected_hand
                item = canvas.create_polygon(
                    points, fill=self.palette["accent"] if selected else "",
                    stipple="gray50" if selected else "",
                    outline=self.palette["accent"] if selected else "",
                    width=2, tags=("hand-finger", hand, finger))
                canvas.tag_bind(item, "<Button-1>",
                                lambda _event, h=hand, f=finger: self._toggle_hand_finger(h, f))
                canvas.tag_bind(item, "<Enter>",
                                lambda _event, i=item: self._finger_hover(i, True))
                canvas.tag_bind(item, "<Leave>",
                                lambda _event, i=item, is_selected=selected:
                                self._finger_hover(i, False, is_selected))

    def _finger_hover(self, item: int, entered: bool, selected: bool = False) -> None:
        self.hand_diagram.itemconfigure(
            item,
            outline=self.palette["text"] if entered else
            (self.palette["accent"] if selected else self.palette["panel"]),
            width=2 if entered or selected else 1)

    def _toggle_hand_finger(self, hand: str, finger: str) -> None:
        self.hand_diagram.focus_set()
        self.hand_var.set(hand)
        self.finger_vars[finger].set(not self.finger_vars[finger].get())
        self._selection_changed()
        self._draw_hand_diagram()

    def _keyboard_select(self, finger: str) -> None:
        hand = self.hand_var.get()
        if hand == "Either":
            hand = "Left"
        self._toggle_hand_finger(hand, finger)

    def _full_frame(self) -> None:
        self.bounds = (0.0, 0.0, 1.0, 1.0)
        for variable, value in zip(self.bound_vars, self.bounds):
            variable.set(value)

    @staticmethod
    def _combo(parent, label, variable, values, command) -> None:
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.pack(fill=tk.X, pady=(2, 8))
        combo.bind("<<ComboboxSelected>>", command)

    def _selection_changed(self, _event=None) -> None:
        selected = self._selected_fingers()
        self.selection_summary.configure(
            text="Selected: " + (", ".join(selected) if selected else "none"))
        if not selected:
            self.status_var.set("Select at least one finger to trace")
            return
        self.tracker.select_many(self.hand_var.get(), tuple(selected))
        self.last_points.clear()
        for trail in self.trails.values():
            trail.break_segment()

    def _selected_fingers(self):
        return [finger for finger in FINGER_NAMES if self.finger_vars[finger].get()]

    def _start_trace(self) -> None:
        if not self._selected_fingers():
            self.status_var.set("Select at least one finger to trace")
            return
        for finger in self._selected_fingers():
            self.trails[finger].start()
        self.status_var.set("Tracing: " + ", ".join(self._selected_fingers()))
        self.trace_state = "running"
        self.primary_button.configure(text="Pause tracing")

    def _primary_trace_action(self) -> None:
        if self.trace_state == "running":
            self._pause_trace()
        else:
            if self.trace_state == "paused":
                self._resume_trace()
            else:
                self._start_trace()

    def _pause_trace(self) -> None:
        for trail in self.trails.values():
            trail.pause()
        self.trace_state = "paused"
        self.primary_button.configure(text="Resume tracing")
        self.status_var.set("Tracing paused")

    def _resume_trace(self) -> None:
        for trail in self.trails.values():
            trail.resume()
        self.trace_state = "running"
        self.primary_button.configure(text="Pause tracing")
        self.status_var.set("Tracing: " + ", ".join(self._selected_fingers()))

    def _stop_trace(self) -> None:
        for trail in self.trails.values():
            trail.stop()
        self.trace_state = "stopped"
        self.primary_button.configure(text="Start tracing")
        self.status_var.set("Ready to trace")

    def _clear_trail(self) -> None:
        for trail in self.trails.values():
            trail.clear()
        self.last_points.clear()

    def _choose_color(self) -> None:
        chosen = colorchooser.askcolor(color=self.trail_color, title="Trail color")[1]
        if chosen:
            self.trail_color = chosen

    def _apply_bounds(self) -> None:
        values = tuple(max(0.0, min(1.0, variable.get())) for variable in self.bound_vars)
        if values[2] <= values[0] or values[3] <= values[1]:
            messagebox.showerror("Invalid tracking area", "Right and bottom must be greater than left and top.")
            return
        self.bounds = values

    def _capture_loop(self) -> None:
        assert self.processor is not None
        try:
            self.processor.open()
            self.root.after(0, lambda: self.connection_var.set("●  Camera connected"))
            while not self.stop_event.is_set():
                item = self.processor.read()
                try:
                    self.frames.put_nowait(item)
                except queue.Full:
                    self.frames.get_nowait()
                    self.frames.put_nowait(item)
        except Exception as error:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait((np.zeros((480, 640, 3), dtype=np.uint8), []))
            self.root.after(0, lambda: (self.connection_var.set("●  Camera error"),
                                        self.status_var.set(str(error))))
        finally:
            if self.processor:
                self.processor.close()

    def _poll_frames(self) -> None:
        try:
            frame, observations = self.frames.get_nowait()
        except queue.Empty:
            self.root.after(30, self._poll_frames)
            return
        if self.processor is None:
            self.root.after(30, self._poll_frames)
            return
        points = self.tracker.update_many(observations)
        for finger in self._selected_fingers():
            point = points.get(finger)
            if point is None:
                if finger in self.last_points:
                    self.trails[finger].break_segment()
                self.last_points.pop(finger, None)
                continue
            mapped = map_point(point, self.bounds, frame.shape[1], frame.shape[0], self.mirror_var.get())
            self.trails[finger].add(mapped)
            self.last_points[finger] = mapped
        if points:
            first = next(iter(points))
            self.position_var.set("%s: %d, %d" % (first, *self.last_points[first]))
            self.status_var.set("Tracking: " + ", ".join(points))
        elif observations:
            self.status_var.set("Hand detected • looking for selected fingers")
        else:
            self.status_var.set("Searching for a hand")
        now = time.monotonic()
        self.fps = 1.0 / max(0.001, now - self.last_frame_time)
        self.last_frame_time = now
        self.fps_var.set("FPS: %.1f" % self.fps)
        self.resolution_var.set("Camera: %d × %d  •  Inference: %.1f ms" %
                                (frame.shape[1], frame.shape[0], self.processor.inference_ms))
        self._render(frame, observations)
        self.root.after(15, self._poll_frames)

    def _render(self, frame, observations) -> None:
        selected_text = ", ".join(self._selected_fingers()) or "No finger selected"
        cv2.rectangle(frame, (16, 16), (330, 52), (11, 13, 18), -1)
        cv2.putText(frame, "TRACE  " + selected_text, (28, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (243, 245, 250), 1, cv2.LINE_AA)
        if self.landmarks_var.get():
            for observation in observations:
                for landmark in observation.landmarks.values():
                    x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (180, 180, 180), -1)
                for finger in self._selected_fingers():
                    tip = observation.fingertip(finger)
                    if tip:
                        active = self.hand_var.get() == "Either" or observation.label == self.hand_var.get()
                        color = (0, 255, 255) if active else (150, 150, 150)
                        cv2.circle(frame, (int(tip[0] * frame.shape[1]), int(tip[1] * frame.shape[0])), 9, color, 2)
        color = tuple(int(self.trail_color[index:index + 2], 16) for index in (5, 3, 1))
        for trail in self.trails.values():
            for segment in trail.segments:
                for first, second in zip(segment, segment[1:]):
                    cv2.line(frame, tuple(map(int, first)), tuple(map(int, second)),
                             color, int(self.thickness.get()), cv2.LINE_AA)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        available_w = max(1, self.preview.winfo_width())
        available_h = max(1, self.preview.winfo_height())
        scale = min(available_w / image.shape[1], available_h / image.shape[0])
        if scale < 1:
            image = cv2.resize(image, (max(1, int(image.shape[1] * scale)),
                                       max(1, int(image.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
        image = ImageTk.PhotoImage(Image.fromarray(image))
        self.preview.configure(image=image)
        self.preview.image = image

    def run(self) -> None:
        try:
            self.processor = CameraProcessor()
        except Exception as error:
            self.status_var.set(str(error))
            self.root.mainloop()
            return
        self.worker = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker.start()
        self.root.mainloop()

    def close(self) -> None:
        self.stop_event.set()
        self.connection_var.set("●  Camera disconnected")
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.0)
        self.root.destroy()
