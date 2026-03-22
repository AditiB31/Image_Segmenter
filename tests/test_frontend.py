"""Tests for frontend JavaScript structure and HTML template."""

import os
import re

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestHTMLTemplate:
    """Test the HTML template has required elements."""

    @pytest.fixture(autouse=True)
    def load_html(self):
        path = os.path.join(BASE_DIR, "templates", "index.html")
        with open(path) as f:
            self.html = f.read()

    def test_file_input_exists(self):
        """Single file input for images exists."""
        assert 'id="file-input"' in self.html

    def test_folder_input_exists(self):
        """Folder input with webkitdirectory exists."""
        assert 'id="folder-input"' in self.html
        assert "webkitdirectory" in self.html

    def test_multi_file_input_exists(self):
        """Multi-file input for images exists."""
        assert 'id="file-input-multiple"' in self.html
        assert "multiple" in self.html

    def test_upload_buttons_exist(self):
        """Select File and Select Folder buttons exist."""
        assert 'id="upload-file-btn"' in self.html
        assert 'id="upload-folder-btn"' in self.html

    def test_pdf_input_exists(self):
        """PDF file input exists."""
        assert 'id="pdf-file-input"' in self.html

    def test_mode_tabs_exist(self):
        """All three mode tabs exist."""
        assert 'data-mode="image"' in self.html
        assert 'data-mode="pdf"' in self.html
        assert 'data-mode="browse"' in self.html

    def test_annotation_tools_exist(self):
        """Annotation tools (point, polygon, box) exist."""
        assert 'data-tool="point"' in self.html
        assert 'data-tool="polygon"' in self.html
        assert 'data-tool="box"' in self.html

    def test_drop_zone_mentions_folder(self):
        """Drop zone mentions folder upload."""
        assert "folder" in self.html.lower()


class TestJavaScript:
    """Test the JavaScript file has required functions and handlers."""

    @pytest.fixture(autouse=True)
    def load_js(self):
        path = os.path.join(BASE_DIR, "static", "js", "app.js")
        with open(path) as f:
            self.js = f.read()

    def test_auto_open_file_picker(self):
        """switchMode should accept autoOpen parameter."""
        assert "autoOpen" in self.js

    def test_mode_tabs_pass_auto_open(self):
        """Mode tab click handlers pass autoOpen=true."""
        assert "switchMode(mode, true)" in self.js

    def test_folder_input_handler(self):
        """Folder input change handler exists."""
        assert "folderInput.addEventListener" in self.js

    def test_handle_image_files_function(self):
        """handleImageFiles function for multi-image upload exists."""
        assert "async function handleImageFiles" in self.js

    def test_render_image_gallery_function(self):
        """renderImageGallery function exists."""
        assert "function renderImageGallery" in self.js

    def test_start_manual_annotate_image_function(self):
        """startManualAnnotateImage function exists."""
        assert "function startManualAnnotateImage" in self.js

    def test_is_image_file_function(self):
        """isImageFile utility function exists and returns boolean."""
        assert "function isImageFile" in self.js
        # Should use .test() not .match() for boolean return
        assert ".test(file.name)" in self.js

    def test_image_gallery_navigation(self):
        """Multi-image navigation in navigateToSlide exists."""
        assert "currentImageList" in self.js

    def test_nav_label_shows_image_prefix(self):
        """Navigation label uses 'Image' prefix for image mode."""
        assert '"Image"' in self.js

    def test_upload_buttons_wired(self):
        """Upload file/folder buttons have click handlers."""
        assert "uploadFileBtn.addEventListener" in self.js
        assert "uploadFolderBtn.addEventListener" in self.js

    def test_get_total_slides_helper(self):
        """getTotalSlides helper function exists to avoid duplication."""
        assert "function getTotalSlides" in self.js

    def test_no_duplicate_keyboard_listeners(self):
        """Only one keydown listener should exist (no duplicates)."""
        count = self.js.count('document.addEventListener("keydown"')
        assert count == 1, f"Expected 1 keydown listener, found {count}"

    def test_no_redundant_state_variable(self):
        """currentImageIndex should not exist (replaced by currentSlideIndex)."""
        assert "currentImageIndex" not in self.js

    def test_file_inputs_reset(self):
        """All file inputs should be reset in resetState."""
        assert 'fileInputMultiple.value = ""' in self.js
        assert 'folderInput.value = ""' in self.js


class TestCSS:
    """Test the CSS file has required styles."""

    @pytest.fixture(autouse=True)
    def load_css(self):
        path = os.path.join(BASE_DIR, "static", "css", "style.css")
        with open(path) as f:
            self.css = f.read()

    def test_upload_buttons_style(self):
        """Upload buttons container has flex styling."""
        assert ".upload-buttons" in self.css
