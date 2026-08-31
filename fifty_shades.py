import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------
# Application appearance
# ---------------------------------------------------------------------

BG = "#1e1e2e"
CARD = "#2a2a3e"
ACCENT = "#89b4fa"
TEXT = "#e5e5f0"
MUTED = "#a0a0bb"

THUMB_MAX = (240, 180)
GRID_COLUMNS = 5

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


# ---------------------------------------------------------------------
# Filter definitions
# ---------------------------------------------------------------------

def generate_filters():
    """
    Return exactly 50 labelled filter definitions.

    The original program created only 49 filters, despite being named
    a "50 Shades" generator.
    """

    filters = []

    for i in range(10):
        filters.append(
            (
                f"Brightness {i + 1}",
                f"brightness({0.5 + i * 0.1:.2f})",
            )
        )

    for i in range(10):
        filters.append(
            (
                f"Contrast {i + 1}",
                f"contrast({0.5 + i * 0.1:.2f})",
            )
        )

    for i in range(10):
        filters.append(
            (
                f"Saturation {i + 1}",
                f"saturate({0.2 + i * 0.18:.2f})",
            )
        )

    for i in range(10):
        filters.append(
            (
                f"Hue {i + 1}",
                f"hue-rotate({i * 36}deg)",
            )
        )

    filters.extend(
        [
            ("Grayscale", "grayscale(100%)"),
            ("Sepia", "sepia(100%)"),
            ("Invert", "invert(100%)"),
            ("Soft Blur", "blur(2px)"),
            ("Opacity", "opacity(0.7)"),
            ("Bright Soft", "brightness(1.2) contrast(0.8)"),
            ("Cool Hue", "hue-rotate(90deg) saturate(1.5)"),
            ("Warm Sepia", "sepia(50%) brightness(1.1) contrast(1.1)"),
            ("Muted Gray", "grayscale(50%) contrast(1.2) saturate(0.8)"),
            ("Dramatic", "saturate(0) brightness(0.8) contrast(1.5)"),
        ]
    )

    return filters


# ---------------------------------------------------------------------
# Image filter engine
# ---------------------------------------------------------------------

def _parse_number(value):
    """Parse a CSS-like numeric value such as 1.2, 50%, or 2px."""

    value = value.strip().lower()

    for suffix in ("deg", "px", "%"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break

    return float(value)


def _percentage(value):
    """Return a CSS percentage as a float between 0 and 1."""

    value = value.strip().lower()

    if value.endswith("%"):
        return _parse_number(value) / 100.0

    return _parse_number(value)


def _hue_rotate(image, degrees):
    """
    Rotate hue while preserving saturation and value.

    The previous version attempted to use PIL's RGB conversion matrix
    with a 16-value matrix. PIL RGB conversion expects a 12-value
    matrix, so hue rotation could fail.
    """

    hsv = image.convert("HSV")
    data = np.asarray(hsv, dtype=np.uint8).copy()

    shift = int(round((degrees % 360) * 255 / 360))
    data[:, :, 0] = (
        data[:, :, 0].astype(np.uint16) + shift
    ) % 256

    return Image.fromarray(data, mode="HSV").convert("RGB")


def _sepia(image):
    """Create a sepia image using a correct RGB transformation."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)

    transformed = np.empty_like(rgb)

    transformed[:, :, 0] = (
        rgb[:, :, 0] * 0.393
        + rgb[:, :, 1] * 0.769
        + rgb[:, :, 2] * 0.189
    )

    transformed[:, :, 1] = (
        rgb[:, :, 0] * 0.349
        + rgb[:, :, 1] * 0.686
        + rgb[:, :, 2] * 0.168
    )

    transformed[:, :, 2] = (
        rgb[:, :, 0] * 0.272
        + rgb[:, :, 1] * 0.534
        + rgb[:, :, 2] * 0.131
    )

    transformed = np.clip(
        transformed,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        transformed,
        mode="RGB",
    )


def apply_css_filter(image, filter_string):
    """
    Apply a CSS-inspired filter sequence.

    Each operation uses the result of the previous operation. Partial
    grayscale, sepia and invert therefore blend against the immediately
    preceding image rather than against the original source.
    """

    image = image.convert("RGB")

    operations = re.findall(
        r"([a-z-]+)\(([^)]*)\)",
        filter_string.lower(),
    )

    for function, argument in operations:

        current = image

        if function == "brightness":

            image = ImageEnhance.Brightness(
                current
            ).enhance(
                _parse_number(argument)
            )

        elif function == "contrast":

            image = ImageEnhance.Contrast(
                current
            ).enhance(
                _parse_number(argument)
            )

        elif function == "saturate":

            image = ImageEnhance.Color(
                current
            ).enhance(
                _parse_number(argument)
            )

        elif function == "hue-rotate":

            image = _hue_rotate(
                current,
                _parse_number(argument),
            )

        elif function == "grayscale":

            amount = max(
                0.0,
                min(
                    1.0,
                    _percentage(argument),
                ),
            )

            gray = ImageOps.grayscale(
                current
            ).convert("RGB")

            image = Image.blend(
                current,
                gray,
                amount,
            )

        elif function == "sepia":

            amount = max(
                0.0,
                min(
                    1.0,
                    _percentage(argument),
                ),
            )

            sepia_image = _sepia(
                current
            )

            image = Image.blend(
                current,
                sepia_image,
                amount,
            )

        elif function == "invert":

            amount = max(
                0.0,
                min(
                    1.0,
                    _percentage(argument),
                ),
            )

            inverted = ImageOps.invert(
                current
            )

            image = Image.blend(
                current,
                inverted,
                amount,
            )

        elif function == "blur":

            radius = max(
                0.0,
                _parse_number(argument),
            )

            image = current.filter(
                ImageFilter.GaussianBlur(
                    radius
                )
            )

        elif function == "opacity":

            amount = max(
                0.0,
                min(
                    1.0,
                    _percentage(argument),
                ),
            )

            white = Image.new(
                "RGB",
                current.size,
                "white",
            )

            image = Image.blend(
                white,
                current,
                amount,
            )

        else:

            raise ValueError(
                f"Unsupported filter: {function}"
            )

    return image.convert("RGB")


# ---------------------------------------------------------------------
# PIL / Qt conversion
# ---------------------------------------------------------------------

def pil_to_qpixmap(image):
    """
    Convert a PIL image to QPixmap safely.

    The QImage owns a copied buffer before this function returns, avoiding
    lifetime issues with a temporary Python bytes object.
    """

    image = image.convert("RGB")

    data = image.tobytes(
        "raw",
        "RGB",
    )

    qimage = QImage(
        data,
        image.width,
        image.height,
        image.width * 3,
        QImage.Format.Format_RGB888,
    ).copy()

    return QPixmap.fromImage(
        qimage
    )


# ---------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------

class PreviewWorker(QThread):

    progress = pyqtSignal(int, int)
    preview_ready = pyqtSignal(int, str, object)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        source_path,
        filters,
    ):

        super().__init__()

        self.source_path = source_path
        self.filters = filters
        self._cancelled = False

    def cancel(self):

        self._cancelled = True

    def run(self):

        try:

            with Image.open(
                self.source_path
            ) as opened:

                source = ImageOps.exif_transpose(
                    opened
                ).convert("RGB")

            preview_source = source.copy()

            preview_source.thumbnail(
                THUMB_MAX,
                Image.Resampling.LANCZOS,
            )

            total = len(
                self.filters
            )

            for index, (
                label,
                filter_string,
            ) in enumerate(
                self.filters
            ):

                if self._cancelled:

                    return

                variant = apply_css_filter(
                    preview_source.copy(),
                    filter_string,
                )

                self.preview_ready.emit(
                    index,
                    label,
                    variant,
                )

                self.progress.emit(
                    index + 1,
                    total,
                )

            self.completed.emit(
                self.source_path
            )

        except Exception as error:

            self.failed.emit(
                str(error)
            )


class ExportWorker(QThread):

    progress = pyqtSignal(int, int)
    completed = pyqtSignal(int, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        source_path,
        filters,
        output_folder,
    ):

        super().__init__()

        self.source_path = source_path
        self.filters = filters
        self.output_folder = Path(
            output_folder
        )

        self._cancelled = False

    def cancel(self):

        self._cancelled = True

    def run(self):

        try:

            self.output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            with Image.open(
                self.source_path
            ) as opened:

                source = ImageOps.exif_transpose(
                    opened
                ).convert("RGB")

            total = len(
                self.filters
            )

            exported = 0

            for index, (
                _label,
                filter_string,
            ) in enumerate(
                self.filters,
                start=1,
            ):

                if self._cancelled:

                    return

                variant = apply_css_filter(
                    source.copy(),
                    filter_string,
                )

                output_path = (
                    self.output_folder
                    / f"shade_{index:02d}.jpg"
                )

                variant.save(
                    output_path,
                    format="JPEG",
                    quality=95,
                    optimize=True,
                )

                exported += 1

                self.progress.emit(
                    index,
                    total,
                )

            self.completed.emit(
                exported,
                str(
                    self.output_folder
                ),
            )

        except Exception as error:

            self.failed.emit(
                str(error)
            )


# ---------------------------------------------------------------------
# Variation card
# ---------------------------------------------------------------------

class VariationCard(QFrame):

    def __init__(
        self,
        label,
        pixmap,
    ):

        super().__init__()

        self.setObjectName(
            "card"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )

        layout.setSpacing(
            6
        )

        image_label = QLabel()

        image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        image_label.setFixedSize(
            *THUMB_MAX
        )

        image_label.setPixmap(
            pixmap
        )

        layout.addWidget(
            image_label
        )

        caption = QLabel(
            label
        )

        caption.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        caption.setStyleSheet(
            f"color: {MUTED}; font-weight: bold;"
        )

        layout.addWidget(
            caption
        )


# ---------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(
        self,
        filters,
    ):

        super().__init__()

        self.filters = filters
        self.source_path = None
        self.preview_worker = None
        self.export_worker = None

        self.setWindowTitle(
            'Image "50 Shades" Generator'
        )

        self.resize(
            1400,
            900,
        )

        self.setMinimumSize(
            900,
            600,
        )

        self.setAcceptDrops(
            True
        )

        self.setStyleSheet(
            self._qss()
        )

        self._build_ui()

    # -----------------------------------------------------------------
    # Appearance
    # -----------------------------------------------------------------

    @staticmethod
    def _qss():

        return f"""
        QMainWindow, QWidget {{
            background: {BG};
            color: {TEXT};
        }}

        QLabel {{
            color: {TEXT};
        }}

        QLabel#title {{
            font-size: 22px;
            font-weight: bold;
        }}

        QLabel#sub {{
            color: {MUTED};
            font-size: 12px;
        }}

        QFrame#card {{
            background: {CARD};
            border-radius: 10px;
        }}

        QPushButton {{
            background: {ACCENT};
            color: #11111b;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
        }}

        QPushButton:hover {{
            background: #74c7ec;
        }}

        QPushButton:disabled {{
            background: #45475a;
            color: {MUTED};
        }}

        QScrollArea {{
            border: 1px solid #313244;
            border-radius: 8px;
            background: {CARD};
        }}

        QProgressBar {{
            border: 1px solid #45475a;
            border-radius: 4px;
            text-align: center;
            background: {CARD};
        }}

        QProgressBar::chunk {{
            background: {ACCENT};
        }}

        QStatusBar {{
            background: {CARD};
            color: {MUTED};
        }}

        QToolBar {{
            background: {CARD};
            spacing: 4px;
        }}
        """

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------

    def _build_ui(self):

        toolbar = self.addToolBar(
            "Actions"
        )

        toolbar.setMovable(
            False
        )

        self.open_action = QAction(
            "Open Image",
            self,
        )

        self.open_action.triggered.connect(
            self.open_image
        )

        toolbar.addAction(
            self.open_action
        )

        self.export_action = QAction(
            "Export All Variations",
            self,
        )

        self.export_action.setEnabled(
            False
        )

        self.export_action.triggered.connect(
            self.export_all
        )

        toolbar.addAction(
            self.export_action
        )

        self.clear_action = QAction(
            "Clear",
            self,
        )

        self.clear_action.triggered.connect(
            self.clear_image
        )

        toolbar.addAction(
            self.clear_action
        )

        central = QWidget()

        self.setCentralWidget(
            central
        )

        root = QVBoxLayout(
            central
        )

        root.setContentsMargins(
            16,
            16,
            16,
            16,
        )

        root.setSpacing(
            12
        )

        title = QLabel(
            'Image "50 Shades" Generator'
        )

        title.setObjectName(
            "title"
        )

        root.addWidget(
            title
        )

        subtitle = QLabel(
            "Open or drag an image to generate 50 filter variations."
        )

        subtitle.setObjectName(
            "sub"
        )

        root.addWidget(
            subtitle
        )

        splitter = QSplitter(
            Qt.Orientation.Horizontal
        )

        # Original image panel.

        left = QWidget()

        left_layout = QVBoxLayout(
            left
        )

        left_layout.setContentsMargins(
            0,
            0,
            8,
            0,
        )

        original_title = QLabel(
            "Original Image"
        )

        original_title.setObjectName(
            "sub"
        )

        left_layout.addWidget(
            original_title
        )

        self.original_label = QLabel(
            "No image loaded."
        )

        self.original_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.original_label.setMinimumSize(
            420,
            480,
        )

        self._reset_original_style()

        left_layout.addWidget(
            self.original_label,
            1,
        )

        # Variations panel.

        right = QWidget()

        right_layout = QVBoxLayout(
            right
        )

        right_layout.setContentsMargins(
            8,
            0,
            0,
            0,
        )

        variations_title = QLabel(
            "50 Shades Variations"
        )

        variations_title.setObjectName(
            "sub"
        )

        right_layout.addWidget(
            variations_title
        )

        self.scroll = QScrollArea()

        self.scroll.setWidgetResizable(
            True
        )

        self.grid_holder = QWidget()

        self.grid = QGridLayout(
            self.grid_holder
        )

        self.grid.setContentsMargins(
            10,
            10,
            10,
            10,
        )

        self.grid.setSpacing(
            10
        )

        for column in range(
            GRID_COLUMNS
        ):

            self.grid.setColumnStretch(
                column,
                1,
            )

        self.scroll.setWidget(
            self.grid_holder
        )

        right_layout.addWidget(
            self.scroll,
            1,
        )

        splitter.addWidget(
            left
        )

        splitter.addWidget(
            right
        )

        splitter.setSizes(
            [
                500,
                900,
            ]
        )

        root.addWidget(
            splitter,
            1,
        )

        self.progress = QProgressBar()

        self.progress.setRange(
            0,
            len(self.filters),
        )

        self.progress.setValue(
            0
        )

        self.progress.setVisible(
            False
        )

        root.addWidget(
            self.progress
        )

        self.statusBar().showMessage(
            "Open an image to begin."
        )

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------

    def open_image(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            (
                "Images (*.png *.jpg *.jpeg *.bmp "
                "*.tif *.tiff *.webp);;"
                "All files (*.*)"
            ),
        )

        if path:

            self.load_image(
                path
            )

    def load_image(
        self,
        path,
    ):

        if self.preview_worker is not None:

            if self.preview_worker.isRunning():

                self.preview_worker.cancel()

                self.preview_worker.wait()

        try:

            with Image.open(
                path
            ) as opened:

                source = ImageOps.exif_transpose(
                    opened
                ).convert("RGB")

                source.thumbnail(
                    (
                        900,
                        700,
                    ),
                    Image.Resampling.LANCZOS,
                )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not open image:\n{error}",
            )

            return

        self.source_path = path

        self._clear_grid()

        self.original_label.setText(
            ""
        )

        self.original_label.setPixmap(
            pil_to_qpixmap(
                source
            )
        )

        self.original_label.setStyleSheet(
            f"background: {CARD}; border-radius: 10px;"
        )

        self.export_action.setEnabled(
            False
        )

        self.progress.setVisible(
            True
        )

        self.progress.setRange(
            0,
            len(self.filters),
        )

        self.progress.setValue(
            0
        )

        self.statusBar().showMessage(
            "Generating preview variations..."
        )

        self.preview_worker = PreviewWorker(
            path,
            self.filters,
        )

        self.preview_worker.preview_ready.connect(
            self._add_preview
        )

        self.preview_worker.progress.connect(
            self._preview_progress
        )

        self.preview_worker.completed.connect(
            self._preview_completed
        )

        self.preview_worker.failed.connect(
            self._preview_failed
        )

        self.preview_worker.start()

    def _add_preview(
        self,
        index,
        label,
        image,
    ):

        pixmap = pil_to_qpixmap(
            image
        ).scaled(
            *THUMB_MAX,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        card = VariationCard(
            label,
            pixmap,
        )

        self.grid.addWidget(
            card,
            index // GRID_COLUMNS,
            index % GRID_COLUMNS,
        )

    def _preview_progress(
        self,
        current,
        total,
    ):

        self.progress.setMaximum(
            total
        )

        self.progress.setValue(
            current
        )

        self.statusBar().showMessage(
            f"Generating preview {current} of {total}..."
        )

    def _preview_completed(
        self,
        path,
    ):

        self.progress.setVisible(
            False
        )

        self.export_action.setEnabled(
            True
        )

        self.statusBar().showMessage(
            f"Loaded {Path(path).name} — "
            f"{len(self.filters)} variations ready."
        )

        self.preview_worker = None

    def _preview_failed(
        self,
        message,
    ):

        self.progress.setVisible(
            False
        )

        self.preview_worker = None

        self.statusBar().showMessage(
            "Variation generation failed."
        )

        QMessageBox.critical(
            self,
            "Filter Error",
            message,
        )

    # -----------------------------------------------------------------
    # Clearing
    # -----------------------------------------------------------------

    def _clear_grid(self):

        while self.grid.count():

            item = self.grid.takeAt(
                0
            )

            widget = item.widget()

            if widget is not None:

                widget.deleteLater()

    def _reset_original_style(self):

        self.original_label.setStyleSheet(
            f"""
            background: {CARD};
            border-radius: 10px;
            color: {MUTED};
            font-size: 16px;
            """
        )

    def clear_image(self):

        if (
            self.preview_worker is not None
            and self.preview_worker.isRunning()
        ):

            self.preview_worker.cancel()

        self.source_path = None

        self._clear_grid()

        self.original_label.clear()

        self.original_label.setText(
            "No image loaded."
        )

        self._reset_original_style()

        self.export_action.setEnabled(
            False
        )

        self.progress.setVisible(
            False
        )

        self.statusBar().showMessage(
            "Cleared."
        )

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    def export_all(self):

        if not self.source_path:

            QMessageBox.information(
                self,
                "Nothing to Export",
                "Load an image first.",
            )

            return

        if (
            self.preview_worker is not None
            and self.preview_worker.isRunning()
        ):

            QMessageBox.information(
                self,
                "Please Wait",
                "Preview generation is still in progress.",
            )

            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose Output Folder",
        )

        if not folder:

            return

        self.export_action.setEnabled(
            False
        )

        self.open_action.setEnabled(
            False
        )

        self.clear_action.setEnabled(
            False
        )

        self.progress.setVisible(
            True
        )

        self.progress.setRange(
            0,
            len(self.filters),
        )

        self.progress.setValue(
            0
        )

        self.statusBar().showMessage(
            "Exporting full-resolution variations..."
        )

        self.export_worker = ExportWorker(
            self.source_path,
            self.filters,
            folder,
        )

        self.export_worker.progress.connect(
            self._export_progress
        )

        self.export_worker.completed.connect(
            self._export_completed
        )

        self.export_worker.failed.connect(
            self._export_failed
        )

        self.export_worker.start()

    def _export_progress(
        self,
        current,
        total,
    ):

        self.progress.setMaximum(
            total
        )

        self.progress.setValue(
            current
        )

        self.statusBar().showMessage(
            f"Exporting {current} of {total}..."
        )

    def _restore_actions(self):

        self.open_action.setEnabled(
            True
        )

        self.clear_action.setEnabled(
            True
        )

        self.export_action.setEnabled(
            self.source_path is not None
        )

    def _export_completed(
        self,
        count,
        folder,
    ):

        self.progress.setVisible(
            False
        )

        self.export_worker = None

        self._restore_actions()

        self.statusBar().showMessage(
            f"Exported {count} variations to {folder}"
        )

        QMessageBox.information(
            self,
            "Export Complete",
            (
                f"Saved {count} full-resolution "
                f"variations to:\n{folder}"
            ),
        )

    def _export_failed(
        self,
        message,
    ):

        self.progress.setVisible(
            False
        )

        self.export_worker = None

        self._restore_actions()

        self.statusBar().showMessage(
            "Export failed."
        )

        QMessageBox.critical(
            self,
            "Export Error",
            message,
        )

    # -----------------------------------------------------------------
    # Drag and drop
    # -----------------------------------------------------------------

    def dragEnterEvent(
        self,
        event,
    ):

        if event.mimeData().hasUrls():

            for url in event.mimeData().urls():

                path = Path(
                    url.toLocalFile()
                )

                if (
                    path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                ):

                    event.acceptProposedAction()

                    return

        event.ignore()

    def dropEvent(
        self,
        event,
    ):

        for url in event.mimeData().urls():

            path = Path(
                url.toLocalFile()
            )

            if (
                path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):

                self.load_image(
                    str(path)
                )

                event.acceptProposedAction()

                return

        event.ignore()

    # -----------------------------------------------------------------
    # Safe shutdown
    # -----------------------------------------------------------------

    def closeEvent(
        self,
        event,
    ):

        workers = (
            self.preview_worker,
            self.export_worker,
        )

        for worker in workers:

            if (
                worker is not None
                and worker.isRunning()
            ):

                worker.cancel()

        for worker in workers:

            if (
                worker is not None
                and worker.isRunning()
            ):

                worker.wait(
                    3000
                )

        event.accept()


# ---------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------

def main():

    QApplication.setStyle(
        QStyleFactory.create(
            "Fusion"
        )
    )

    app = QApplication(
        sys.argv
    )

    app.setFont(
        QFont(
            "Helvetica",
            10,
        )
    )

    window = MainWindow(
        generate_filters()
    )

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":

    main()
