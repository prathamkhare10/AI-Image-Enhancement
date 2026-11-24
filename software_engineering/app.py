
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import time
import torch

from modules.image_processor import load_image
from modules.model_service import ModelService
from modules.file_handler import save_image
from modules.evaluation import evaluate_images


class ZoomViewer:

    def __init__(self, parent, image, title):

        self.image = image.copy()
        self.window = tk.Toplevel(parent)

        self.window.title(title + " - Image Viewer")
        self.window.geometry("1000x750")
        self.window.configure(bg="#f1f5f9")

        self.zoom = 1.0
        self.photo = None
        self.render_job = None

        # Top controls
        controls = tk.Frame(
            self.window,
            bg="#f1f5f9",
            padx=10,
            pady=10
        )
        controls.pack(fill=tk.X)

        tk.Button(
            controls,
            text="Fit to Window",
            command=self.fit_image
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls,
            text="Actual Size (100%)",
            command=lambda: self.set_zoom(1.0)
        ).pack(side=tk.LEFT, padx=5)

        self.zoom_label = tk.Label(
            controls,
            text="Zoom: 100%",
            bg="#f1f5f9",
            font=("Arial", 10, "bold")
        )
        self.zoom_label.pack(side=tk.LEFT, padx=15)

        self.zoom_slider = tk.Scale(
            controls,
            from_=0.05,
            to=3.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=280,
            showvalue=False,
            command=self.change_zoom,
            bg="#f1f5f9"
        )
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(side=tk.LEFT, padx=5)

        # Viewer area
        viewer = tk.Frame(self.window, bg="#dbeafe")
        viewer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(
            viewer,
            bg="#ffffff",
            highlightthickness=0
        )

        vertical_scroll = tk.Scrollbar(
            viewer,
            orient=tk.VERTICAL,
            command=self.canvas.yview
        )

        horizontal_scroll = tk.Scrollbar(
            viewer,
            orient=tk.HORIZONTAL,
            command=self.canvas.xview
        )

        self.canvas.configure(
            yscrollcommand=vertical_scroll.set,
            xscrollcommand=horizontal_scroll.set
        )

        vertical_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        horizontal_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Drag to move around the zoomed image
        self.canvas.bind(
            "<ButtonPress-1>",
            self.start_pan
        )

        self.canvas.bind(
            "<B1-Motion>",
            self.pan_image
        )

        # Mouse wheel scrolling
        self.canvas.bind(
            "<MouseWheel>",
            self.mouse_scroll
        )

        # Ctrl + mouse wheel to zoom
        self.canvas.bind(
            "<Control-MouseWheel>",
            self.mouse_zoom
        )

        self.canvas.bind(
            "<Configure>",
            self.schedule_render
        )

        self.window.after(150, self.fit_image)

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def pan_image(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def mouse_scroll(self, event):
        self.canvas.yview_scroll(
            int(-event.delta / 120),
            "units"
        )

    def mouse_zoom(self, event):

        if event.delta > 0:
            new_zoom = min(3.0, self.zoom + 0.1)
        else:
            new_zoom = max(0.05, self.zoom - 0.1)

        self.set_zoom(new_zoom)

    def schedule_render(self, event=None):

        if self.render_job is not None:
            self.window.after_cancel(self.render_job)

        self.render_job = self.window.after(
            100,
            self.render_image
        )

    def set_zoom(self, value):

        self.zoom = max(0.05, min(3.0, float(value)))

        self.zoom_slider.set(self.zoom)
        self.zoom_label.config(
            text=f"Zoom: {self.zoom * 100:.0f}%"
        )

        self.schedule_render()

    def change_zoom(self, value):

        self.zoom = float(value)

        self.zoom_label.config(
            text=f"Zoom: {self.zoom * 100:.0f}%"
        )

        self.schedule_render()

    def fit_image(self):

        self.window.update_idletasks()

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)

        image_width, image_height = self.image.size

        fit_scale = min(
            canvas_width / image_width,
            canvas_height / image_height
        )

        self.set_zoom(min(3.0, max(0.05, fit_scale)))

    def render_image(self):

        self.render_job = None

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)

        image_width, image_height = self.image.size

        new_width = max(1, int(image_width * self.zoom))
        new_height = max(1, int(image_height * self.zoom))

        resized = self.image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        self.photo = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")

        # Center the image when it is smaller than the canvas
        x = max(0, (canvas_width - new_width) // 2)
        y = max(0, (canvas_height - new_height) // 2)

        self.canvas.create_image(
            x,
            y,
            anchor=tk.NW,
            image=self.photo
        )

        self.canvas.configure(
            scrollregion=(
                0,
                0,
                max(canvas_width, x + new_width),
                max(canvas_height, y + new_height)
            )
        )


class ImageEnhancementApp:

    def __init__(self, root):

        self.root = root
        self.root.title("AI Image Enhancement System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f1f5f9")

        self.original_image = None
        self.enhanced_image = None
        self.reference_image = None
        self.original_path = None

        self.model_service = None

        self.original_photo = None
        self.enhanced_photo = None

        self.create_dashboard()

    def create_dashboard(self):

        tk.Label(
            self.root,
            text="AI-Based Image Enhancement System",
            font=("Arial", 23, "bold"),
            bg="#f1f5f9",
            fg="#172554"
        ).pack(pady=(18, 4))

        tk.Label(
            self.root,
            text="Image Super-Resolution using SRCNN",
            font=("Arial", 12),
            bg="#f1f5f9",
            fg="#64748b"
        ).pack()

        buttons = tk.Frame(self.root, bg="#f1f5f9")
        buttons.pack(pady=16)

        self.create_button(
            buttons, "Upload Image",
            self.upload_image, "#2563eb"
        ).pack(side=tk.LEFT, padx=6)

        self.create_button(
            buttons, "Enhance Image",
            self.enhance_image, "#16a34a"
        ).pack(side=tk.LEFT, padx=6)

        self.create_button(
            buttons, "Load HR Reference",
            self.load_reference, "#0f766e"
        ).pack(side=tk.LEFT, padx=6)

        self.create_button(
            buttons, "Save Enhanced Image",
            self.save_enhanced_image, "#7c3aed"
        ).pack(side=tk.LEFT, padx=6)

        # Image preview panels
        previews = tk.Frame(self.root, bg="#f1f5f9")
        previews.pack(
            fill=tk.BOTH,
            expand=True,
            padx=18,
            pady=8
        )

        original_frame = tk.LabelFrame(
            previews,
            text="Original Image",
            font=("Arial", 12, "bold"),
            bg="white",
            padx=8,
            pady=8
        )
        original_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=6
        )

        enhanced_frame = tk.LabelFrame(
            previews,
            text="Enhanced Image",
            font=("Arial", 12, "bold"),
            bg="white",
            padx=8,
            pady=8
        )
        enhanced_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=6
        )

        self.original_label = tk.Label(
            original_frame,
            text="Upload an image to preview",
            bg="white"
        )
        self.original_label.pack(expand=True)

        self.enhanced_label = tk.Label(
            enhanced_frame,
            text="Enhanced image will appear here",
            bg="white"
        )
        self.enhanced_label.pack(expand=True)

        self.original_label.bind(
            "<Double-Button-1>",
            lambda event: self.open_zoom(
                self.original_image,
                "Original Image"
            )
        )

        self.enhanced_label.bind(
            "<Double-Button-1>",
            lambda event: self.open_zoom(
                self.enhanced_image,
                "Enhanced Image"
            )
        )

        # Dimensions and processing information
        info = tk.Frame(
            self.root,
            bg="white",
            padx=12,
            pady=12
        )
        info.pack(fill=tk.X, padx=24, pady=5)

        self.original_info = tk.Label(
            info,
            text="Original dimensions: --",
            bg="white"
        )
        self.original_info.pack(side=tk.LEFT, padx=12)

        self.enhanced_info = tk.Label(
            info,
            text="Enhanced dimensions: --",
            bg="white"
        )
        self.enhanced_info.pack(side=tk.LEFT, padx=12)

        self.time_info = tk.Label(
            info,
            text="Processing time: --",
            bg="white"
        )
        self.time_info.pack(side=tk.RIGHT, padx=12)

        # Quality evaluation panel
        metrics = tk.LabelFrame(
            self.root,
            text="Image Quality Evaluation",
            font=("Arial", 12, "bold"),
            bg="white",
            padx=15,
            pady=10
        )
        metrics.pack(fill=tk.X, padx=24, pady=8)

        self.psnr_label = tk.Label(
            metrics,
            text="PSNR: Not evaluated",
            font=("Arial", 11, "bold"),
            bg="white"
        )
        self.psnr_label.pack(side=tk.LEFT, padx=35)

        self.ssim_label = tk.Label(
            metrics,
            text="SSIM: Not evaluated",
            font=("Arial", 11, "bold"),
            bg="white"
        )
        self.ssim_label.pack(side=tk.LEFT, padx=35)

        self.reference_info = tk.Label(
            metrics,
            text="Reference image: Not loaded",
            bg="white",
            fg="#64748b"
        )
        self.reference_info.pack(side=tk.RIGHT, padx=20)

        self.status_label = tk.Label(
            self.root,
            text="Status: Ready",
            font=("Arial", 11),
            bg="#f1f5f9",
            fg="#334155"
        )
        self.status_label.pack(pady=8)

        tk.Label(
            self.root,
            text="Double-click an image to zoom. Use Ctrl + mouse wheel to zoom, scrollbars or drag to navigate.",
            font=("Arial", 9),
            bg="#f1f5f9",
            fg="#64748b"
        ).pack(pady=(0, 8))

    def create_button(self, parent, text, command, color):

        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 10, "bold"),
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            padx=12,
            pady=9,
            cursor="hand2",
            relief=tk.FLAT
        )

    def show_preview(self, image, label, side):

        preview = image.copy()
        preview.thumbnail((500, 400))

        photo = ImageTk.PhotoImage(preview)

        label.configure(image=photo, text="")
        label.image = photo

        if side == "original":
            self.original_photo = photo
        else:
            self.enhanced_photo = photo

    def reset_metrics(self):

        self.psnr_label.config(text="PSNR: Not evaluated")
        self.ssim_label.config(text="SSIM: Not evaluated")

    def upload_image(self):

        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png")
            ]
        )

        if not path:
            return

        try:
            self.original_image = load_image(path)
            self.original_path = path
            self.enhanced_image = None
            self.reference_image = None

            self.show_preview(
                self.original_image,
                self.original_label,
                "original"
            )

            self.enhanced_label.config(
                image="",
                text="Click Enhance Image to process",
                bg="white"
            )
            self.enhanced_label.image = None
            self.enhanced_photo = None

            width, height = self.original_image.size

            self.original_info.config(
                text=f"Original dimensions: {width} × {height}"
            )

            self.enhanced_info.config(
                text="Enhanced dimensions: --"
            )

            self.time_info.config(
                text="Processing time: --"
            )

            self.reference_info.config(
                text="Reference image: Not loaded"
            )

            self.reset_metrics()

            self.status_label.config(
                text="Status: Image uploaded successfully"
            )

        except Exception as error:
            messagebox.showerror("Image Error", str(error))

    def enhance_image(self):

        if self.original_image is None:
            messagebox.showwarning(
                "No Image",
                "Please upload an image first."
            )
            return

        try:
            self.status_label.config(
                text="Status: Enhancing image..."
            )
            self.root.update_idletasks()

            if self.model_service is None:
                self.model_service = ModelService()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            self.enhanced_image = (
                self.model_service.enhance_image(
                    self.original_image
                )
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            self.show_preview(
                self.enhanced_image,
                self.enhanced_label,
                "enhanced"
            )

            width, height = self.enhanced_image.size

            self.enhanced_info.config(
                text=f"Enhanced dimensions: {width} × {height}"
            )

            self.time_info.config(
                text=f"Processing time: {elapsed:.2f} seconds"
            )

            self.reset_metrics()

            self.status_label.config(
                text="Status: Image enhanced successfully"
            )

            if self.reference_image is not None:
                self.calculate_metrics()

        except Exception as error:
            messagebox.showerror(
                "Enhancement Error",
                str(error)
            )

            self.status_label.config(
                text="Status: Enhancement failed"
            )

    def load_reference(self):

        path = filedialog.askopenfilename(
            title="Select High-Resolution Reference Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png")
            ]
        )

        if not path:
            return

        try:
            reference = load_image(path)

            if self.enhanced_image is not None:
                if reference.size != self.enhanced_image.size:
                    raise ValueError(
                        "Reference image dimensions must match "
                        "the enhanced image exactly.\n\n"
                        f"Enhanced: {self.enhanced_image.size}\n"
                        f"Reference: {reference.size}"
                    )

            self.reference_image = reference

            self.reference_info.config(
                text=f"Reference image: {reference.width} × {reference.height}"
            )

            if self.enhanced_image is not None:
                self.calculate_metrics()
            else:
                self.status_label.config(
                    text="Status: Reference loaded. Enhance the image to evaluate."
                )

        except Exception as error:
            self.reference_image = None
            self.reference_info.config(
                text="Reference image: Not loaded"
            )
            self.reset_metrics()
            messagebox.showerror(
                "Reference Image Error",
                str(error)
            )

    def calculate_metrics(self):

        if self.enhanced_image is None or self.reference_image is None:
            return

        try:
            results = evaluate_images(
                self.enhanced_image,
                self.reference_image
            )

            self.psnr_label.config(
                text=f"PSNR: {results['psnr']:.2f} dB"
            )

            self.ssim_label.config(
                text=f"SSIM: {results['ssim']:.4f}"
            )

            self.status_label.config(
                text="Status: Quality evaluation completed"
            )

        except Exception as error:
            self.reset_metrics()
            messagebox.showerror(
                "Evaluation Error",
                str(error)
            )

    def save_enhanced_image(self):

        if self.enhanced_image is None:
            messagebox.showwarning(
                "No Enhanced Image",
                "Please enhance an image first."
            )
            return

        try:
            path = save_image(
                self.enhanced_image,
                self.original_path
            )

            messagebox.showinfo(
                "Image Saved",
                f"Enhanced image saved successfully:\n{path}"
            )

            self.status_label.config(
                text="Status: Enhanced image saved successfully"
            )

        except Exception as error:
            messagebox.showerror(
                "Save Error",
                str(error)
            )

    def open_zoom(self, image, title):

        if image is None:
            return

        ZoomViewer(
            self.root,
            image,
            title
        )


if __name__ == "__main__":

    root = tk.Tk()
    app = ImageEnhancementApp(root)
    root.mainloop()
