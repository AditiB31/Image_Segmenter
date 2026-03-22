// ── DOM Elements ──────────────────────────────────────────────────────
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileInputMultiple = document.getElementById("file-input-multiple");
const folderInput = document.getElementById("folder-input");
const uploadFileBtn = document.getElementById("upload-file-btn");
const uploadFolderBtn = document.getElementById("upload-folder-btn");
const pdfDropZone = document.getElementById("pdf-drop-zone");
const pdfFileInput = document.getElementById("pdf-file-input");
const uploadSection = document.getElementById("upload-section");
const pdfUploadSection = document.getElementById("pdf-upload-section");
const browseSection = document.getElementById("browse-section");
const slidesSection = document.getElementById("slides-section");
const processingSection = document.getElementById("processing-section");
const resultsSection = document.getElementById("results-section");
const statusText = document.getElementById("status-text");
const statusHint = document.getElementById("status-hint");
const segmentsGrid = document.getElementById("segments-grid");
const segmentCount = document.getElementById("segment-count");
const visibleCount = document.getElementById("visible-count");
const downloadBtn = document.getElementById("download-btn");
const selectAllBtn = document.getElementById("select-all-btn");
const clearSelectionBtn = document.getElementById("clear-selection-btn");
const selectedCount = document.getElementById("selected-count");
const startOverBtn = document.getElementById("start-over-btn");
const areaFilter = document.getElementById("area-filter");
const areaValue = document.getElementById("area-value");
const sortBy = document.getElementById("sort-by");
const previewImage = document.getElementById("preview-image");
const upscaleSelect = document.getElementById("upscale-select");

// Modal elements
const previewModal = document.getElementById("preview-modal");
const modalClose = document.getElementById("modal-close");
const modalImage = document.getElementById("modal-image");
const modalInfo = document.getElementById("modal-info");
const modalDownload = document.getElementById("modal-download");

// Mode tabs
const modeTabs = document.querySelectorAll(".mode-tab");

// Slides section
const slidesGrid = document.getElementById("slides-grid");
const slidesTitle = document.getElementById("slides-title");
const slidesSubtitle = document.getElementById("slides-subtitle");
const backToSourceBtn = document.getElementById("back-to-source-btn");

// Results navigation
const backToSlidesBtn = document.getElementById("back-to-slides-btn");
const slideNav = document.getElementById("slide-nav");
const prevSlideBtn = document.getElementById("prev-slide-btn");
const nextSlideBtn = document.getElementById("next-slide-btn");
const slideNavLabel = document.getElementById("slide-nav-label");

// Browse section
const runsList = document.getElementById("runs-list");
const browseLoading = document.getElementById("browse-loading");
const browseEmpty = document.getElementById("browse-empty");

// Mode choice section
const modeChoiceSection = document.getElementById("mode-choice-section");

// Annotation section
const annotateSection = document.getElementById("annotate-section");
const annotateCanvas = document.getElementById("annotate-canvas");

// ── State ─────────────────────────────────────────────────────────────
let currentMode = "image";       // "image" | "pdf" | "browse"
let currentSessionId = null;
let allSegments = [];
let selectedIndices = new Set();
let segmentMap = null;           // cached Map(index → segment) for sorting
let lastClickedCardIdx = null;   // grid position of last clicked card (for shift-click range)

// PDF state
let currentPdfSlides = [];
let currentSlideIndex = null;
let currentSlideName = null;
let segmentedSlides = new Set();  // track which slides have been segmented

// Browse state
let currentRunName = null;
let currentRunSlides = [];
let browseImagesDir = null;

// Multi-image state (folder upload)
let currentImageList = [];     // [{session_id, filename, width, height, name}]

// Annotation state
let annotateImage = null;
let annotateScale = 1;
let annotateTool = "point";
let currentPrompt = { points: [], contour: [], contourClosed: false, box: null };
let extractedSegments = [];
let boxDragStart = null;
let currentUploadData = null;
let annotateSlideIndex = null;  // non-null when annotating a PDF slide

// ── Toast Notifications ───────────────────────────────────────────────
const toastContainer = document.getElementById("toast-container");

function showToast(message, type = "error", duration = 5000) {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.classList.add("toast-out");
        toast.addEventListener("animationend", () => toast.remove());
    }, duration);
}

// Track source name for download filenames
let currentSourceName = "";

// ── Utilities ─────────────────────────────────────────────────────────
async function getErrorMessage(res, fallback) {
    try {
        const data = await res.json();
        return data.error || fallback;
    } catch {
        return `${fallback} (${res.status})`;
    }
}

function debounce(fn, ms) {
    let timer;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

// ── Icons ─────────────────────────────────────────────────────────────
const downloadIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;
const checkIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;
const zoomIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>`;

function getUpscale() {
    return parseInt(upscaleSelect.value);
}

function getTotalSlides() {
    if (currentMode === "browse") return currentRunSlides.length;
    if (currentMode === "image") return currentImageList.length;
    return currentPdfSlides.length;
}

// ── URL Helpers (mode-aware) ──────────────────────────────────────────
function getThumbUrl(seg) {
    if (currentMode === "browse") {
        return `/browse/thumb/${currentRunName}/${currentSlideName}/${seg.filename}`;
    }
    if (currentMode === "pdf" && currentSlideName) {
        return `/segment-image/${currentSessionId}/${seg.filename}?slide=${currentSlideName}`;
    }
    return `/segment-image/${currentSessionId}/${seg.filename}`;
}

function getDownloadUrl(seg) {
    if (currentMode === "browse") {
        return `/browse/download/${currentRunName}/${currentSlideName}/${seg.filename}`;
    }
    const slide = currentMode === "pdf" && currentSlideName ? `&slide=${currentSlideName}` : "";
    return `/download/${currentSessionId}/${seg.filename}?upscale=${getUpscale()}${slide}`;
}

// ── Mode Switching ────────────────────────────────────────────────────
modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        const mode = tab.dataset.mode;
        switchMode(mode, true);
    });
});

function switchMode(mode, autoOpen = false) {
    currentMode = mode;
    modeTabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));

    // Reset state
    resetState();

    if (mode === "image") {
        showSection("upload");
        if (autoOpen) setTimeout(() => fileInputMultiple.click(), 100);
    } else if (mode === "pdf") {
        showSection("pdf-upload");
        if (autoOpen) setTimeout(() => pdfFileInput.click(), 100);
    } else if (mode === "browse") {
        showSection("browse");
        loadRuns();
    }
}

function resetState() {
    segmentsGrid.innerHTML = "";
    allSegments = [];
    selectedIndices.clear();
    segmentMap = null;
    currentSessionId = null;
    currentPdfSlides = [];
    currentSlideIndex = null;
    currentSlideName = null;
    currentRunName = null;
    currentRunSlides = [];
    browseImagesDir = null;
    segmentedSlides.clear();
    currentSourceName = "";
    extractedSegments = [];
    currentUploadData = null;
    currentImageList = [];
    annotateImage = null;
    annotateSlideIndex = null;
    currentPrompt = { points: [], contour: [], contourClosed: false, box: null };
    boxDragStart = null;
    fileInput.value = "";
    fileInputMultiple.value = "";
    folderInput.value = "";
    pdfFileInput.value = "";
    previewImage.hidden = true;
    previewImage.src = "";
    sortBy.value = "area-desc";
    upscaleSelect.value = "2";
}

// ── Image Drag & Drop ─────────────────────────────────────────────────
function isImageFile(file) {
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
    return validTypes.includes(file.type) || /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name);
}

dropZone.addEventListener("click", (e) => {
    // Don't trigger file picker if a button inside was clicked
    if (e.target.closest(".upload-buttons")) return;
    fileInput.click();
});
uploadFileBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInputMultiple.click(); });
uploadFolderBtn.addEventListener("click", (e) => { e.stopPropagation(); folderInput.click(); });
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = [...e.dataTransfer.files].filter(isImageFile);
    if (files.length > 1) {
        handleImageFiles(files);
    } else if (files.length === 1) {
        handleImageFile(files[0]);
    }
});
fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleImageFile(fileInput.files[0]);
});
fileInputMultiple.addEventListener("change", () => {
    const files = [...fileInputMultiple.files].filter(isImageFile);
    if (files.length > 1) {
        handleImageFiles(files);
    } else if (files.length === 1) {
        handleImageFile(files[0]);
    }
});
folderInput.addEventListener("change", () => {
    const files = [...folderInput.files].filter(isImageFile);
    if (files.length > 1) {
        handleImageFiles(files);
    } else if (files.length === 1) {
        handleImageFile(files[0]);
    } else {
        showToast("No image files found in the selected folder.");
    }
});

// ── PDF Drag & Drop ───────────────────────────────────────────────────
pdfDropZone.addEventListener("click", () => pdfFileInput.click());
pdfDropZone.addEventListener("dragover", (e) => { e.preventDefault(); pdfDropZone.classList.add("drag-over"); });
pdfDropZone.addEventListener("dragleave", () => pdfDropZone.classList.remove("drag-over"));
pdfDropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    pdfDropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handlePdfFile(file);
});
pdfFileInput.addEventListener("change", () => {
    if (pdfFileInput.files[0]) handlePdfFile(pdfFileInput.files[0]);
});

// ── Start Over ────────────────────────────────────────────────────────
startOverBtn.addEventListener("click", () => {
    if (currentMode === "pdf" && currentPdfSlides.length > 0) {
        showSection("slides");
        return;
    }
    if (currentMode === "browse") {
        showSection("browse");
        loadRuns();
        return;
    }
    if (currentMode === "image" && currentImageList.length > 1) {
        renderImageGallery();
        return;
    }
    if (currentMode === "image" && currentUploadData) {
        startManualAnnotate();
        return;
    }
    switchMode(currentMode);
});

// ── Back to slides ────────────────────────────────────────────────────
backToSlidesBtn.addEventListener("click", () => {
    if (currentMode === "image" && currentImageList.length > 1) {
        renderImageGallery();
    } else {
        showSection("slides");
    }
});

// ── Back to source (from slides view) ─────────────────────────────────
backToSourceBtn.addEventListener("click", () => {
    if (currentMode === "browse") {
        showSection("browse");
        loadRuns();
    } else if (currentMode === "pdf") {
        showSection("pdf-upload");
    } else {
        showSection("upload");
    }
});

// ── Slide Navigation ──────────────────────────────────────────────────
prevSlideBtn.addEventListener("click", () => {
    if (currentSlideIndex > 0) navigateToSlide(currentSlideIndex - 1);
});
nextSlideBtn.addEventListener("click", () => {
    const totalSlides = getTotalSlides();
    if (currentSlideIndex < totalSlides - 1) navigateToSlide(currentSlideIndex + 1);
});

function navigateToSlide(idx) {
    if (currentMode === "browse") {
        const slide = currentRunSlides[idx];
        loadBrowseSlideSegments(currentRunName, slide.name, idx);
    } else if (currentMode === "pdf") {
        startManualAnnotateSlide(idx);
    } else if (currentMode === "image" && currentImageList.length > 1) {
        startManualAnnotateImage(idx);
    }
}

// ── Area filter ───────────────────────────────────────────────────────
const debouncedFilter = debounce(filterSegments, 50);
areaFilter.addEventListener("input", () => {
    areaValue.textContent = parseInt(areaFilter.value).toLocaleString() + " px";
    debouncedFilter(parseInt(areaFilter.value));
});

// ── Sort handler ──────────────────────────────────────────────────────
sortBy.addEventListener("change", () => sortSegments(sortBy.value));

// ── Select all / clear ────────────────────────────────────────────────
selectAllBtn.addEventListener("click", () => {
    const visibleCards = segmentsGrid.querySelectorAll('.segment-card:not([style*="display: none"])');
    visibleCards.forEach((card) => {
        selectedIndices.add(parseInt(card.dataset.index));
        card.classList.add("selected");
    });
    updateSelectionUI();
});

clearSelectionBtn.addEventListener("click", () => {
    selectedIndices.clear();
    segmentsGrid.querySelectorAll(".segment-card.selected").forEach((card) => card.classList.remove("selected"));
    updateSelectionUI();
});

// ── Keyboard shortcuts ────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        if (previewModal.classList.contains("active")) { closeModal(); return; }
        if (!resultsSection.hidden) clearSelectionBtn.click();
        return;
    }
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (!resultsSection.hidden && (e.ctrlKey || e.metaKey) && e.key === "a") {
        e.preventDefault();
        selectAllBtn.click();
    }
});

// ── Download ──────────────────────────────────────────────────────────
downloadBtn.addEventListener("click", async () => {
    if (currentMode === "browse") {
        // For browse mode, download rendered segments directly
        await downloadBrowseSegments();
        return;
    }

    const indices = selectedIndices.size > 0 ? [...selectedIndices] : allSegments.map((s) => s.index);
    const upscale = getUpscale();
    downloadBtn.disabled = true;
    downloadBtn.textContent = "Preparing...";

    try {
        const payload = { indices, upscale };
        if (currentMode === "pdf" && currentSlideName) payload.slide = currentSlideName;
        const res = await fetch(`/download-all/${currentSessionId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await getErrorMessage(res, "Download failed"));

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const prefix = currentSourceName || "segments";
        const slideSuffix = currentSlideName ? `_${currentSlideName}` : "";
        a.download = `${prefix}${slideSuffix}_${upscale}x.zip`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        showToast("Error: " + err.message);
    } finally {
        downloadBtn.disabled = false;
        updateSelectionUI();
    }
});

async function downloadBrowseSegments() {
    // Download rendered PNGs from browse mode as a ZIP
    const indices = selectedIndices.size > 0 ? [...selectedIndices] : allSegments.map((s) => s.index);
    downloadBtn.disabled = true;
    downloadBtn.textContent = "Preparing ZIP...";

    try {
        const res = await fetch(`/browse/download-all/${currentRunName}/${currentSlideName}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ indices }),
        });
        if (!res.ok) throw new Error(await getErrorMessage(res, "Download failed"));

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const runPrefix = currentSourceName || currentRunName || "browse";
        a.download = `${runPrefix}_${currentSlideName}_segments.zip`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        showToast("Error: " + err.message);
    } finally {
        downloadBtn.disabled = false;
        updateSelectionUI();
    }
}

// ── Modal ─────────────────────────────────────────────────────────────
function openModal(seg) {
    modalImage.src = getThumbUrl(seg);
    modalInfo.innerHTML = `
        <strong>Segment ${seg.index}</strong><br>
        ${seg.width} x ${seg.height} px &middot;
        ${seg.area.toLocaleString()} px&sup2; &middot;
        IoU ${seg.predicted_iou}
    `;
    modalDownload.onclick = () => { window.location.href = getDownloadUrl(seg); };
    previewModal.classList.add("active");
}

function closeModal() { previewModal.classList.remove("active"); }
modalClose.addEventListener("click", closeModal);
previewModal.addEventListener("click", (e) => { if (e.target === previewModal) closeModal(); });

// ── Image File Handling ───────────────────────────────────────────────
async function handleImageFile(file) {
    if (!isImageFile(file)) {
        showToast("Unsupported file type. Please upload JPG, PNG, WebP, BMP, or TIFF.");
        return;
    }
    if (file.size > 50 * 1024 * 1024) { showToast("File is too large. Maximum size is 50 MB."); return; }

    const previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    previewImage.hidden = false;
    previewImage.onload = () => URL.revokeObjectURL(previewUrl);

    showSection("processing");
    statusText.textContent = "Uploading image...";
    statusHint.textContent = "";

    try {
        const formData = new FormData();
        formData.append("file", file);
        const uploadRes = await fetch("/upload", { method: "POST", body: formData });
        if (!uploadRes.ok) throw new Error(await getErrorMessage(uploadRes, "Upload failed"));
        const uploadData = await uploadRes.json();
        currentSessionId = uploadData.session_id;
        currentSourceName = uploadData.filename.replace(/\.[^.]+$/, "");
        currentUploadData = uploadData;

        // Go directly to manual annotation (default workflow)
        startManualAnnotate();
    } catch (err) {
        showSection("upload");
        showToast("Error: " + err.message);
    }
}

// ── Multiple Image File Handling (folder upload) ─────────────────────
async function handleImageFiles(files) {
    showSection("processing");
    statusText.textContent = `Uploading ${files.length} images...`;
    statusHint.textContent = "";

    const uploaded = [];
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.size > 50 * 1024 * 1024) {
            showToast(`Skipping ${file.name}: exceeds 50 MB limit.`);
            continue;
        }
        statusText.textContent = `Uploading ${i + 1} of ${files.length}...`;
        try {
            const formData = new FormData();
            formData.append("file", file);
            const res = await fetch("/upload", { method: "POST", body: formData });
            if (!res.ok) {
                showToast(`Failed to upload ${file.name}`);
                continue;
            }
            const data = await res.json();
            uploaded.push({
                session_id: data.session_id,
                filename: data.filename,
                name: data.filename.replace(/\.[^.]+$/, ""),
                width: data.width,
                height: data.height,
            });
        } catch (err) {
            showToast(`Error uploading ${file.name}: ${err.message}`);
        }
    }

    if (uploaded.length === 0) {
        showSection("upload");
        showToast("No images were uploaded successfully.");
        return;
    }

    currentImageList = uploaded;
    currentSourceName = uploaded.length === 1 ? uploaded[0].name : "images";
    renderImageGallery();
}

function renderImageGallery() {
    const slides = currentImageList.map((img) => ({
        name: img.name,
        filename: img.filename,
        width: img.width,
        height: img.height,
        _session_id: img.session_id,
    }));
    renderSlidesGrid(`${currentImageList.length} Images`, slides, false);
}

function startManualAnnotateImage(idx) {
    const img = currentImageList[idx];
    currentSlideIndex = idx;
    currentSessionId = img.session_id;
    currentSourceName = img.name;
    currentUploadData = img;
    annotateSlideIndex = null;
    extractedSegments = [];
    currentPrompt = { points: [], contour: [], contourClosed: false, box: null };
    boxDragStart = null;
    showSection("annotate");
    setupAnnotateCanvas(`/original-image/${img.session_id}`);
    renderAnnotateSegments();
}

// ── PDF File Handling ─────────────────────────────────────────────────
async function handlePdfFile(file) {
    if (!file.name.match(/\.pdf$/i)) {
        showToast("Please upload a PDF file.");
        return;
    }
    if (file.size > 50 * 1024 * 1024) { showToast("File is too large. Maximum size is 50 MB."); return; }

    showSection("processing");
    statusText.textContent = "Uploading PDF...";
    statusHint.textContent = "Converting slides to images";

    try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch("/upload-pdf", { method: "POST", body: formData });
        if (!res.ok) throw new Error(await getErrorMessage(res, "Upload failed"));
        const data = await res.json();

        currentSessionId = data.session_id;
        currentPdfSlides = data.slides;
        currentSourceName = data.pdf_name;
        segmentedSlides.clear();

        renderSlidesGrid(data.pdf_name, data.slides, data.cached);
    } catch (err) {
        showSection("pdf-upload");
        showToast("Error: " + err.message);
    }
}

// ── Slides Grid (shared by PDF upload and browse) ─────────────────────
function renderSlidesGrid(title, slides, cached) {
    showSection("slides");
    slidesTitle.textContent = title;
    slidesSubtitle.textContent = cached
        ? `${slides.length} slides (cached)`
        : `${slides.length} slides`;

    slidesGrid.innerHTML = "";

    slides.forEach((slide, idx) => {
        const card = document.createElement("div");
        card.className = "slide-card";
        card.dataset.index = idx;
        card.dataset.name = slide.name;

        const isSegmented = segmentedSlides.has(slide.name);
        const hasSegments = slide.segment_count !== undefined && slide.segment_count > 0;
        if (isSegmented || hasSegments) card.classList.add("segmented");

        let thumbSrc;
        if (currentMode === "image" && slide._session_id) {
            thumbSrc = `/original-image/${slide._session_id}`;
        } else if (currentMode === "pdf") {
            thumbSrc = `/slide-image/${currentSessionId}/${slide.filename || slide.name + ".png"}`;
        } else if (currentMode === "browse" && currentRunName) {
            thumbSrc = `/browse/slide-image/${currentRunName}/${slide.name}`;
        } else {
            thumbSrc = "";
        }

        const badge = isSegmented || hasSegments
            ? `<div class="slide-badge">${checkIcon} ${slide.segment_count || ""} segments</div>`
            : `<div class="slide-badge slide-badge-action">Click to annotate</div>`;

        card.innerHTML = `
            <div class="slide-preview">
                ${thumbSrc ? `<img src="${thumbSrc}" alt="${slide.name}" loading="lazy">` : `<div class="slide-placeholder">${slide.name}</div>`}
                ${badge}
            </div>
            <div class="slide-info">
                <span class="slide-name">${slide.name.replace(/_/g, " ")}</span>
                ${slide.width ? `<span class="slide-dims">${slide.width} x ${slide.height}</span>` : ""}
            </div>
        `;

        card.addEventListener("click", () => {
            if (currentMode === "image") {
                startManualAnnotateImage(idx);
            } else if (currentMode === "pdf") {
                startManualAnnotateSlide(idx);
            } else if (currentMode === "browse") {
                loadBrowseSlideSegments(currentRunName, slide.name, idx);
            }
        });

        slidesGrid.appendChild(card);
    });
}

// ── Browse Runs ───────────────────────────────────────────────────────
async function loadRuns() {
    browseLoading.hidden = false;
    browseEmpty.hidden = true;
    runsList.hidden = true;

    try {
        const res = await fetch("/browse/runs");
        if (!res.ok) throw new Error(`Server error (${res.status})`);
        const data = await res.json();
        browseLoading.hidden = true;

        if (data.runs.length === 0) {
            browseEmpty.hidden = false;
            return;
        }

        runsList.hidden = false;
        runsList.innerHTML = "";

        data.runs.forEach((run) => {
            const card = document.createElement("div");
            card.className = "run-card";

            const ts = run.timestamp
                ? new Date(
                    run.timestamp.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, "$1-$2-$3T$4:$5:$6")
                  ).toLocaleString()
                : "";

            card.innerHTML = `
                <div class="run-info">
                    <strong class="run-name">${run.pdf_name.replace(/_/g, " ")}</strong>
                    <span class="run-meta">
                        ${run.slide_count} slide${run.slide_count !== 1 ? "s" : ""} &middot;
                        ${run.total_segments} segments
                        ${ts ? ` &middot; ${ts}` : ""}
                    </span>
                    ${run.settings && run.settings.upscale ? `<span class="run-settings">${run.settings.upscale}x upscale &middot; DPI ${run.settings.dpi || "?"}</span>` : ""}
                </div>
                <button class="btn btn-primary btn-small">View</button>
            `;

            card.addEventListener("click", () => loadRun(run.name));
            runsList.appendChild(card);
        });
    } catch (err) {
        browseLoading.hidden = true;
        showToast("Error loading runs: " + err.message);
    }
}

async function loadRun(runName) {
    currentRunName = runName;
    showSection("processing");
    statusText.textContent = "Loading run...";
    statusHint.textContent = "";

    try {
        const res = await fetch(`/browse/run/${runName}`);
        if (!res.ok) throw new Error(`Server error (${res.status})`);
        const data = await res.json();

        currentRunSlides = data.slides;
        browseImagesDir = data.images_dir;
        currentSourceName = data.pdf_name || runName;

        // Build slides data for the grid
        const slides = data.slides.map((s) => ({
            name: s.name,
            segment_count: s.segment_count,
            has_rendered: s.has_rendered,
        }));

        renderSlidesGrid(data.pdf_name.replace(/_/g, " "), slides, false);
    } catch (err) {
        showSection("browse");
        showToast("Error loading run: " + err.message);
    }
}

async function loadBrowseSlideSegments(runName, slideName, slideIdx) {
    currentSlideIndex = slideIdx;
    currentSlideName = slideName;

    showSection("processing");
    statusText.textContent = `Loading ${slideName.replace(/_/g, " ")}...`;
    statusHint.textContent = "";

    try {
        const res = await fetch(`/browse/run/${runName}/${slideName}`);
        if (!res.ok) throw new Error(`Server error (${res.status})`);
        const data = await res.json();
        allSegments = data.segments;
        selectedIndices.clear();
        renderResults();
    } catch (err) {
        showSection("slides");
        showToast("Error: " + err.message);
    }
}

// ── Render Results ────────────────────────────────────────────────────
function renderResults() {
    showSection("results");
    segmentCount.textContent = `${allSegments.length} segments found`;

    // Show/hide slide navigation and mode-specific controls
    const hasSlides = currentMode === "pdf" || currentMode === "browse" || (currentMode === "image" && currentImageList.length > 1);
    backToSlidesBtn.hidden = !hasSlides;

    // Hide upscale control in browse mode (pre-rendered at fixed upscale)
    const upscaleControl = document.querySelector(".upscale-control");
    if (upscaleControl) upscaleControl.hidden = currentMode === "browse";

    if (hasSlides && currentSlideIndex !== null) {
        slideNav.hidden = false;
        const totalSlides = getTotalSlides();
        const navPrefix = currentMode === "image" ? "Image" : "Slide";
        slideNavLabel.textContent = `${navPrefix} ${currentSlideIndex + 1} / ${totalSlides}`;
        slideNavLabel.style.cursor = "pointer";
        slideNavLabel.title = currentMode === "image" ? "Back to images" : "Back to slides grid";
        slideNavLabel.onclick = () => {
            if (currentMode === "image" && currentImageList.length > 1) {
                renderImageGallery();
            } else {
                showSection("slides");
            }
        };
        prevSlideBtn.disabled = currentSlideIndex <= 0;
        nextSlideBtn.disabled = currentSlideIndex >= totalSlides - 1;
    } else {
        slideNav.hidden = true;
    }

    segmentMap = new Map(allSegments.map((s) => [s.index, s]));

    if (allSegments.length > 0) {
        const maxArea = Math.max(...allSegments.map((s) => s.area));
        areaFilter.max = Math.floor(maxArea / 2);
        areaFilter.value = 0;
        areaValue.textContent = "0 px";
    }

    segmentsGrid.innerHTML = "";

    if (allSegments.length === 0) {
        segmentsGrid.innerHTML = `<div class="empty-state"><p>No segments found for this slide.</p></div>`;
        updateSelectionUI();
        return;
    }

    allSegments.forEach((seg) => {
        const card = document.createElement("div");
        card.className = "segment-card";
        card.dataset.area = seg.area;
        card.dataset.index = seg.index;

        card.innerHTML = `
            <div class="segment-preview">
                <div class="selection-indicator">${checkIcon}</div>
                <img src="${getThumbUrl(seg)}" alt="Segment ${seg.index}" loading="lazy">
            </div>
            <div class="segment-info">
                <div class="segment-meta">
                    ${seg.width} x ${seg.height} px<br>
                    ${seg.area.toLocaleString()} px area
                </div>
                <div class="segment-actions">
                    <button class="segment-zoom" title="Preview">${zoomIcon}</button>
                    <button class="segment-download" title="Download full-res PNG">${downloadIcon}</button>
                </div>
            </div>
        `;

        card.addEventListener("click", (e) => {
            if (e.target.closest(".segment-download") || e.target.closest(".segment-zoom")) return;
            const cards = [...segmentsGrid.querySelectorAll(".segment-card")];
            const cardIdx = cards.indexOf(card);
            if (e.shiftKey && lastClickedCardIdx !== null) {
                // Range select: toggle all visible cards between last click and this one
                const lo = Math.min(lastClickedCardIdx, cardIdx);
                const hi = Math.max(lastClickedCardIdx, cardIdx);
                for (let i = lo; i <= hi; i++) {
                    if (!cards[i].hidden) {
                        const idx = parseInt(cards[i].dataset.index, 10);
                        selectedIndices.add(idx);
                        cards[i].classList.add("selected");
                    }
                }
                updateSelectionUI();
            } else {
                toggleSelection(card, seg.index);
            }
            lastClickedCardIdx = cardIdx;
        });

        card.querySelector(".segment-zoom").addEventListener("click", (e) => {
            e.stopPropagation();
            openModal(seg);
        });

        card.querySelector(".segment-download").addEventListener("click", (e) => {
            e.stopPropagation();
            window.location.href = getDownloadUrl(seg);
        });

        segmentsGrid.appendChild(card);
    });

    updateSelectionUI();
    updateVisibleCount();
}

// ── Selection & Filtering ─────────────────────────────────────────────
function toggleSelection(card, index) {
    if (selectedIndices.has(index)) {
        selectedIndices.delete(index);
        card.classList.remove("selected");
    } else {
        selectedIndices.add(index);
        card.classList.add("selected");
    }
    updateSelectionUI();
}

function updateSelectionUI() {
    const n = selectedIndices.size;
    const upscale = getUpscale();
    if (n === 0) {
        selectedCount.textContent = "";
        downloadBtn.textContent = currentMode === "browse" ? "Download All" : `Download All ${upscale}x ZIP`;
    } else {
        selectedCount.textContent = `${n} selected`;
        downloadBtn.textContent = currentMode === "browse" ? `Download ${n} Selected` : `Download ${n} Selected ${upscale}x ZIP`;
    }
    downloadBtn.disabled = allSegments.length === 0;
}

upscaleSelect.addEventListener("change", updateSelectionUI);

function filterSegments(minArea) {
    const cards = segmentsGrid.querySelectorAll(".segment-card");
    cards.forEach((card) => {
        card.style.display = parseInt(card.dataset.area) >= minArea ? "" : "none";
    });
    updateVisibleCount();
}

function updateVisibleCount() {
    const total = segmentsGrid.querySelectorAll(".segment-card").length;
    const visible = segmentsGrid.querySelectorAll('.segment-card:not([style*="display: none"])').length;
    visibleCount.textContent = visible < total ? `(showing ${visible} of ${total})` : "";
}

function sortSegments(criterion) {
    if (!segmentMap) return;
    const cards = [...segmentsGrid.querySelectorAll(".segment-card")];
    cards.sort((a, b) => {
        const aSeg = segmentMap.get(parseInt(a.dataset.index));
        const bSeg = segmentMap.get(parseInt(b.dataset.index));
        if (!aSeg || !bSeg) return 0;
        switch (criterion) {
            case "area-desc": return bSeg.area - aSeg.area;
            case "area-asc": return aSeg.area - bSeg.area;
            case "quality": return (bSeg.predicted_iou || 0) - (aSeg.predicted_iou || 0);
            default: return 0;
        }
    });
    cards.forEach((card) => segmentsGrid.appendChild(card));
}

// ── Section Switching ─────────────────────────────────────────────────
function showSection(name) {
    uploadSection.hidden = name !== "upload";
    modeChoiceSection.hidden = name !== "mode-choice";
    pdfUploadSection.hidden = name !== "pdf-upload";
    browseSection.hidden = name !== "browse";
    slidesSection.hidden = name !== "slides";
    processingSection.hidden = name !== "processing";
    annotateSection.hidden = name !== "annotate";
    resultsSection.hidden = name !== "results";
}

// ── Auto Segment (accessible from annotation toolbar) ────────────────

async function autoSegmentCurrent() {
    showSection("processing");

    if (annotateSlideIndex !== null) {
        // Auto-segment a PDF slide
        const slide = currentPdfSlides[annotateSlideIndex];
        statusText.textContent = `Segmenting ${slide.name.replace(/_/g, " ")}...`;
        statusHint.textContent = "This may take a moment";

        try {
            const segRes = await fetch(`/segment-slide/${currentSessionId}/${annotateSlideIndex}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ min_area: 100 }),
            });
            if (!segRes.ok) throw new Error(await getErrorMessage(segRes, "Segmentation failed"));
            const segData = await segRes.json();
            allSegments = segData.segments;
            selectedIndices.clear();
            currentSlideName = slide.name;
            currentSlideIndex = annotateSlideIndex;
            segmentedSlides.add(slide.name);
            currentPdfSlides[annotateSlideIndex].segment_count = segData.segment_count;
            renderResults();
        } catch (err) {
            showSection("annotate");
            showToast("Error: " + err.message);
        }
    } else {
        // Auto-segment an uploaded image
        statusText.textContent = "Segmenting image...";
        statusHint.textContent = currentUploadData
            ? `${currentUploadData.width} \u00d7 ${currentUploadData.height} px \u2014 this may take a moment`
            : "";

        try {
            const segRes = await fetch(`/segment/${currentSessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ min_area: 100 }),
            });
            if (!segRes.ok) throw new Error(await getErrorMessage(segRes, "Segmentation failed"));
            const segData = await segRes.json();
            allSegments = segData.segments;
            selectedIndices.clear();
            currentSlideName = null;
            renderResults();
        } catch (err) {
            showSection("annotate");
            showToast("Error: " + err.message);
        }
    }
}

document.getElementById("annotate-auto-segment-btn").addEventListener("click", autoSegmentCurrent);

// ── Manual Annotation ─────────────────────────────────────────────────

function startManualAnnotate() {
    annotateSlideIndex = null;
    extractedSegments = [];
    currentPrompt = { points: [], contour: [], contourClosed: false, box: null };
    boxDragStart = null;
    showSection("annotate");
    setupAnnotateCanvas(`/original-image/${currentSessionId}`);
    renderAnnotateSegments();
}

function startManualAnnotateSlide(idx) {
    const slide = currentPdfSlides[idx];
    annotateSlideIndex = idx;
    currentSlideIndex = idx;
    currentSlideName = slide.name;
    extractedSegments = [];
    currentPrompt = { points: [], contour: [], contourClosed: false, box: null };
    boxDragStart = null;
    showSection("annotate");
    setupAnnotateCanvas(`/original-slide-image/${currentSessionId}/${idx}`);
    renderAnnotateSegments();
}

function setupAnnotateCanvas(imageSrc) {
    const img = new Image();
    img.onload = () => {
        annotateImage = img;
        resizeAnnotateCanvas();
        redrawAnnotateCanvas();
    };
    img.onerror = () => {
        showToast("Failed to load image for annotation.");
        if (annotateSlideIndex !== null) {
            showSection("slides");
        } else {
            switchMode("image");
        }
    };
    img.src = imageSrc;
}

window.addEventListener("resize", debounce(() => {
    if (!annotateSection.hidden && annotateImage) {
        resizeAnnotateCanvas();
        redrawAnnotateCanvas();
    }
}, 150));

function resizeAnnotateCanvas() {
    if (!annotateImage) return;
    const wrap = document.querySelector(".annotate-canvas-wrap");
    const maxW = wrap.clientWidth - 32 || 900;
    const maxH = window.innerHeight * 0.55;
    const imgW = annotateImage.naturalWidth;
    const imgH = annotateImage.naturalHeight;
    annotateScale = Math.min(maxW / imgW, maxH / imgH, 1);
    annotateCanvas.width = Math.round(imgW * annotateScale);
    annotateCanvas.height = Math.round(imgH * annotateScale);
}

function redrawAnnotateCanvas() {
    if (!annotateImage) return;
    const ctx = annotateCanvas.getContext("2d");
    const w = annotateCanvas.width;
    const h = annotateCanvas.height;
    const s = annotateScale;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(annotateImage, 0, 0, w, h);

    // Draw points
    currentPrompt.points.forEach((pt) => {
        const px = pt.x * s, py = pt.y * s;
        // Outer glow for visibility on any background
        ctx.beginPath();
        ctx.arc(px, py, 10, 0, Math.PI * 2);
        ctx.fillStyle = pt.label === 1 ? "rgba(0, 200, 0, 0.25)" : "rgba(255, 50, 50, 0.25)";
        ctx.fill();
        // Main circle
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.fillStyle = pt.label === 1 ? "rgba(0, 200, 0, 0.9)" : "rgba(255, 50, 50, 0.9)";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
        // Label
        ctx.fillStyle = "#fff";
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(pt.label === 1 ? "+" : "\u2212", px, py + 1);
    });

    // Draw contour
    if (currentPrompt.contour.length > 0) {
        ctx.beginPath();
        ctx.moveTo(currentPrompt.contour[0].x * s, currentPrompt.contour[0].y * s);
        for (let i = 1; i < currentPrompt.contour.length; i++) {
            ctx.lineTo(currentPrompt.contour[i].x * s, currentPrompt.contour[i].y * s);
        }
        if (currentPrompt.contourClosed) {
            ctx.closePath();
            ctx.fillStyle = "rgba(0, 113, 227, 0.15)";
            ctx.fill();
        }
        // Outer stroke for visibility on dark backgrounds
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
        ctx.lineWidth = 4;
        ctx.stroke();
        // Inner coloured stroke
        ctx.strokeStyle = "rgba(0, 113, 227, 0.9)";
        ctx.lineWidth = 2;
        ctx.stroke();

        currentPrompt.contour.forEach((pt, i) => {
            const px = pt.x * s, py = pt.y * s;
            ctx.beginPath();
            ctx.arc(px, py, i === 0 ? 7 : 5, 0, Math.PI * 2);
            ctx.fillStyle = i === 0 ? "rgba(0, 113, 227, 0.95)" : "rgba(0, 113, 227, 0.7)";
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }

    // Draw box
    if (currentPrompt.box) {
        const b = currentPrompt.box;
        const bx = b.x1 * s, by = b.y1 * s, bw = (b.x2 - b.x1) * s, bh = (b.y2 - b.y1) * s;
        ctx.fillStyle = "rgba(0, 113, 227, 0.1)";
        ctx.fillRect(bx, by, bw, bh);
        // Outer white stroke for visibility
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
        ctx.lineWidth = 4;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(bx, by, bw, bh);
        // Inner coloured stroke
        ctx.strokeStyle = "rgba(0, 113, 227, 0.9)";
        ctx.lineWidth = 2;
        ctx.strokeRect(bx, by, bw, bh);
        ctx.setLineDash([]);
    }
}

// ── Canvas Mouse Events ───────────────────────────────────────────────

annotateCanvas.addEventListener("click", (e) => {
    if (annotateTool === "point") {
        const origX = e.offsetX / annotateScale;
        const origY = e.offsetY / annotateScale;
        currentPrompt.points.push({ x: origX, y: origY, label: 1 });
        redrawAnnotateCanvas();
    } else if (annotateTool === "polygon") {
        if (currentPrompt.contourClosed) return;
        const origX = e.offsetX / annotateScale;
        const origY = e.offsetY / annotateScale;
        // Close polygon when clicking near first vertex
        if (currentPrompt.contour.length >= 3) {
            const first = currentPrompt.contour[0];
            const displayDist = Math.hypot((first.x - origX) * annotateScale, (first.y - origY) * annotateScale);
            if (displayDist < 20) {
                currentPrompt.contourClosed = true;
                redrawAnnotateCanvas();
                return;
            }
        }
        currentPrompt.contour.push({ x: origX, y: origY });
        redrawAnnotateCanvas();
    }
});

annotateCanvas.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    if (annotateTool === "point") {
        const origX = e.offsetX / annotateScale;
        const origY = e.offsetY / annotateScale;
        currentPrompt.points.push({ x: origX, y: origY, label: 0 });
        redrawAnnotateCanvas();
    }
});

// Box tool: drag to draw
annotateCanvas.addEventListener("mousedown", (e) => {
    if (annotateTool !== "box" || e.button !== 0) return;
    boxDragStart = { x: e.offsetX / annotateScale, y: e.offsetY / annotateScale };
    currentPrompt.box = null;
});

annotateCanvas.addEventListener("mousemove", (e) => {
    if (annotateTool !== "box" || !boxDragStart) return;
    const x = e.offsetX / annotateScale;
    const y = e.offsetY / annotateScale;
    currentPrompt.box = {
        x1: Math.min(boxDragStart.x, x),
        y1: Math.min(boxDragStart.y, y),
        x2: Math.max(boxDragStart.x, x),
        y2: Math.max(boxDragStart.y, y),
    };
    redrawAnnotateCanvas();
});

annotateCanvas.addEventListener("mouseup", (e) => {
    if (annotateTool !== "box") return;
    boxDragStart = null;
});

// ── Annotation Tool Switching ─────────────────────────────────────────

document.querySelectorAll(".annotate-tool").forEach((btn) => {
    btn.addEventListener("click", () => {
        annotateTool = btn.dataset.tool;
        document.querySelectorAll(".annotate-tool").forEach((b) =>
            b.classList.toggle("active", b === btn)
        );
        updateAnnotateHint();
    });
});

function updateAnnotateHint() {
    const hint = document.getElementById("annotate-tool-hint");
    switch (annotateTool) {
        case "point":
            hint.textContent = "Left-click: foreground point \u00b7 Right-click: background point";
            break;
        case "polygon":
            hint.textContent = "Click to add vertices \u00b7 Click near first point to close polygon";
            break;
        case "box":
            hint.textContent = "Click and drag to draw a bounding box";
            break;
    }
}

// ── Undo / Clear ──────────────────────────────────────────────────────

document.getElementById("annotate-undo-btn").addEventListener("click", () => {
    if (annotateTool === "point" && currentPrompt.points.length > 0) {
        currentPrompt.points.pop();
    } else if (annotateTool === "polygon") {
        if (currentPrompt.contourClosed) {
            currentPrompt.contourClosed = false;
        } else if (currentPrompt.contour.length > 0) {
            currentPrompt.contour.pop();
        }
    } else if (annotateTool === "box") {
        currentPrompt.box = null;
    }
    redrawAnnotateCanvas();
});

document.getElementById("annotate-clear-btn").addEventListener("click", () => {
    currentPrompt = { points: [], contour: [], contourClosed: false, box: null };
    redrawAnnotateCanvas();
});

// ── Extract Segment ───────────────────────────────────────────────────

document.getElementById("annotate-extract-btn").addEventListener("click", extractSegment);

async function extractSegment() {
    const prompt = {};

    if (currentPrompt.points.length > 0) {
        prompt.points = currentPrompt.points.map((p) => [p.x, p.y]);
        prompt.labels = currentPrompt.points.map((p) => p.label);
    }

    if (currentPrompt.contour.length >= 3) {
        prompt.contour = currentPrompt.contour.map((p) => [p.x, p.y]);
    }

    if (currentPrompt.box) {
        prompt.box = [currentPrompt.box.x1, currentPrompt.box.y1, currentPrompt.box.x2, currentPrompt.box.y2];
    }

    if (!prompt.points && !prompt.contour && !prompt.box) {
        showToast("Add annotations first (click points, draw polygon, or draw box).");
        return;
    }

    // Validate box is large enough
    if (prompt.box && !prompt.points && !prompt.contour) {
        const bw = Math.abs(prompt.box[2] - prompt.box[0]);
        const bh = Math.abs(prompt.box[3] - prompt.box[1]);
        if (bw < 5 || bh < 5) {
            showToast("Box too small. Draw a larger area.");
            return;
        }
    }

    const extractBtn = document.getElementById("annotate-extract-btn");
    extractBtn.disabled = true;
    extractBtn.textContent = "Extracting...";

    try {
        const annotateUrl = annotateSlideIndex !== null
            ? `/annotate-slide/${currentSessionId}/${annotateSlideIndex}`
            : `/annotate/${currentSessionId}`;
        const res = await fetch(annotateUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompts: [prompt] }),
        });
        if (!res.ok) throw new Error(await getErrorMessage(res, "Extraction failed"));
        const data = await res.json();

        if (data.segments.length === 0) {
            showToast("No segment found. Try different annotations.");
            return;
        }

        extractedSegments.push(...data.segments);
        currentPrompt = { points: [], contour: [], contourClosed: false, box: null };
        redrawAnnotateCanvas();
        renderAnnotateSegments();
        showToast(`Extracted ${data.segments.length} segment(s)`, "success", 2000);
    } catch (err) {
        showToast("Error: " + err.message);
    } finally {
        extractBtn.disabled = false;
        extractBtn.textContent = "Extract Segment";
    }
}

// ── Annotation Segments Gallery ───────────────────────────────────────

function renderAnnotateSegments() {
    const grid = document.getElementById("annotate-segments-grid");
    const countEl = document.getElementById("annotate-segment-count");
    const doneBtn = document.getElementById("annotate-done-btn");
    const dlBtn = document.getElementById("annotate-download-btn");

    countEl.textContent = `${extractedSegments.length} segment${extractedSegments.length !== 1 ? "s" : ""} extracted`;
    doneBtn.disabled = extractedSegments.length === 0;
    dlBtn.disabled = extractedSegments.length === 0;

    grid.innerHTML = "";
    extractedSegments.forEach((seg, i) => {
        const card = document.createElement("div");
        card.className = "annotate-segment-card";
        card.innerHTML = `
            <img src="/segment-image/${currentSessionId}/${seg.filename}${annotateSlideIndex !== null ? `?slide=${currentSlideName}` : ""}" alt="Segment ${seg.index}">
            <button class="annotate-segment-remove" title="Remove">&times;</button>
        `;
        card.querySelector(".annotate-segment-remove").addEventListener("click", (e) => {
            e.stopPropagation();
            extractedSegments.splice(i, 1);
            renderAnnotateSegments();
        });
        grid.appendChild(card);
    });
}

// ── Done / Back ───────────────────────────────────────────────────────

document.getElementById("annotate-done-btn").addEventListener("click", () => {
    allSegments = [...extractedSegments];
    selectedIndices.clear();
    // currentSlideName is already set for PDF slides, null for images
    renderResults();
});

document.getElementById("annotate-back-btn").addEventListener("click", () => {
    if (annotateSlideIndex !== null) {
        showSection("slides");
    } else if (currentImageList.length > 1) {
        renderImageGallery();
    } else {
        switchMode("image");
    }
});

// ── Annotation Download ───────────────────────────────────────────────

document.getElementById("annotate-download-btn").addEventListener("click", async () => {
    const indices = extractedSegments.map((s) => s.index);
    const upscale = parseInt(document.getElementById("annotate-upscale-select").value);
    const dlBtn = document.getElementById("annotate-download-btn");
    dlBtn.disabled = true;
    dlBtn.textContent = "Preparing...";

    try {
        const payload = { indices, upscale };
        if (annotateSlideIndex !== null && currentSlideName) payload.slide = currentSlideName;
        const res = await fetch(`/download-all/${currentSessionId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await getErrorMessage(res, "Download failed"));

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const slideSuffix = currentSlideName ? `_${currentSlideName}` : "";
        a.download = `${currentSourceName || "segments"}${slideSuffix}_manual_${upscale}x.zip`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        showToast("Error: " + err.message);
    } finally {
        dlBtn.disabled = false;
        dlBtn.textContent = "Download All ZIP";
    }
});
