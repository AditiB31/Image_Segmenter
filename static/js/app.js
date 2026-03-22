// ── DOM Elements ──────────────────────────────────────────────────────
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
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

// ── State ─────────────────────────────────────────────────────────────
let currentMode = "image";       // "image" | "pdf" | "browse"
let currentSessionId = null;
let allSegments = [];
let selectedIndices = new Set();

// PDF state
let currentPdfSlides = [];
let currentSlideIndex = null;
let currentSlideName = null;
let segmentedSlides = new Set();  // track which slides have been segmented

// Browse state
let currentRunName = null;
let currentRunSlides = [];
let browseImagesDir = null;

// ── Icons ─────────────────────────────────────────────────────────────
const downloadIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;
const checkIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;
const zoomIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>`;

function getUpscale() {
    return parseInt(upscaleSelect.value);
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
        switchMode(mode);
    });
});

function switchMode(mode) {
    currentMode = mode;
    modeTabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));

    // Reset state
    resetState();

    if (mode === "image") {
        showSection("upload");
    } else if (mode === "pdf") {
        showSection("pdf-upload");
    } else if (mode === "browse") {
        showSection("browse");
        loadRuns();
    }
}

function resetState() {
    segmentsGrid.innerHTML = "";
    allSegments = [];
    selectedIndices.clear();
    currentSessionId = null;
    currentPdfSlides = [];
    currentSlideIndex = null;
    currentSlideName = null;
    currentRunName = null;
    currentRunSlides = [];
    browseImagesDir = null;
    segmentedSlides.clear();
    fileInput.value = "";
    pdfFileInput.value = "";
    previewImage.hidden = true;
    previewImage.src = "";
    sortBy.value = "area-desc";
    upscaleSelect.value = "2";
}

// ── Image Drag & Drop ─────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleImageFile(file);
});
fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleImageFile(fileInput.files[0]);
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
        // Go back to slides view
        showSection("slides");
        return;
    }
    if (currentMode === "browse") {
        showSection("browse");
        loadRuns();
        return;
    }
    switchMode(currentMode);
});

// ── Back to slides ────────────────────────────────────────────────────
backToSlidesBtn.addEventListener("click", () => {
    showSection("slides");
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
    const totalSlides = currentMode === "browse" ? currentRunSlides.length : currentPdfSlides.length;
    if (currentSlideIndex < totalSlides - 1) navigateToSlide(currentSlideIndex + 1);
});

function navigateToSlide(idx) {
    if (currentMode === "browse") {
        const slide = currentRunSlides[idx];
        loadBrowseSlideSegments(currentRunName, slide.name, idx);
    } else if (currentMode === "pdf") {
        segmentSlide(idx);
    }
}

// ── Area filter ───────────────────────────────────────────────────────
areaFilter.addEventListener("input", () => {
    const minArea = parseInt(areaFilter.value);
    areaValue.textContent = minArea.toLocaleString() + " px";
    filterSegments(minArea);
});

// ── Sort handler ──────────────────────────────────────────────────────
sortBy.addEventListener("change", () => sortSegments(sortBy.value));

// ── Keyboard shortcuts ────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        if (previewModal.classList.contains("active")) { closeModal(); return; }
        if (!resultsSection.hidden) clearSelectionBtn.click();
    }
    if (!resultsSection.hidden && (e.ctrlKey || e.metaKey) && e.key === "a") {
        e.preventDefault();
        selectAllBtn.click();
    }
});

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
        if (!res.ok) { const err = await res.json(); throw new Error(err.error || "Download failed"); }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `segments_${upscale}x.zip`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        downloadBtn.disabled = false;
        updateSelectionUI();
    }
});

async function downloadBrowseSegments() {
    // Download rendered PNGs from browse mode (already rendered by pipeline)
    const indices = selectedIndices.size > 0 ? [...selectedIndices] : allSegments.map((s) => s.index);
    downloadBtn.disabled = true;
    downloadBtn.textContent = "Downloading...";

    try {
        // Download each segment individually (they're already rendered)
        for (const idx of indices) {
            const seg = allSegments.find((s) => s.index === idx);
            if (!seg) continue;
            const url = `/browse/download/${currentRunName}/${currentSlideName}/${seg.filename}`;
            const a = document.createElement("a");
            a.href = url;
            a.download = seg.filename;
            a.click();
            // Small delay to avoid browser throttling
            await new Promise((r) => setTimeout(r, 200));
        }
    } catch (err) {
        alert("Error: " + err.message);
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
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpe?g|png|webp|bmp|tiff?)$/i)) {
        alert("Unsupported file type. Please upload JPG, PNG, WebP, BMP, or TIFF.");
        return;
    }
    if (file.size > 50 * 1024 * 1024) { alert("File is too large. Maximum size is 50 MB."); return; }

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
        if (!uploadRes.ok) { const err = await uploadRes.json(); throw new Error(err.error || "Upload failed"); }
        const uploadData = await uploadRes.json();
        currentSessionId = uploadData.session_id;

        statusText.textContent = "Segmenting image...";
        statusHint.textContent = `${uploadData.width} x ${uploadData.height} px -- this may take a moment`;

        const segRes = await fetch(`/segment/${currentSessionId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ min_area: 100 }),
        });
        if (!segRes.ok) { const err = await segRes.json(); throw new Error(err.error || "Segmentation failed"); }
        const segData = await segRes.json();
        allSegments = segData.segments;
        selectedIndices.clear();
        currentSlideName = null;
        renderResults();
    } catch (err) {
        showSection("upload");
        alert("Error: " + err.message);
    }
}

// ── PDF File Handling ─────────────────────────────────────────────────
async function handlePdfFile(file) {
    if (!file.name.match(/\.pdf$/i)) {
        alert("Please upload a PDF file.");
        return;
    }
    if (file.size > 50 * 1024 * 1024) { alert("File is too large. Maximum size is 50 MB."); return; }

    showSection("processing");
    statusText.textContent = "Uploading PDF...";
    statusHint.textContent = "Converting slides to images";

    try {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch("/upload-pdf", { method: "POST", body: formData });
        if (!res.ok) { const err = await res.json(); throw new Error(err.error || "Upload failed"); }
        const data = await res.json();

        currentSessionId = data.session_id;
        currentPdfSlides = data.slides;
        segmentedSlides.clear();

        renderSlidesGrid(data.pdf_name, data.slides, data.cached);
    } catch (err) {
        showSection("pdf-upload");
        alert("Error: " + err.message);
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
        if (currentMode === "pdf") {
            thumbSrc = `/slide-image/${currentSessionId}/${slide.filename || slide.name + ".png"}`;
        } else {
            // Browse mode: try to load from cached images
            thumbSrc = "";
        }

        const badge = isSegmented || hasSegments
            ? `<div class="slide-badge">${checkIcon} ${slide.segment_count || ""} segments</div>`
            : `<div class="slide-badge slide-badge-action">Click to segment</div>`;

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
            if (currentMode === "pdf") {
                segmentSlide(idx);
            } else if (currentMode === "browse") {
                loadBrowseSlideSegments(currentRunName, slide.name, idx);
            }
        });

        slidesGrid.appendChild(card);
    });
}

// ── Segment a PDF Slide ───────────────────────────────────────────────
async function segmentSlide(idx) {
    currentSlideIndex = idx;
    const slide = currentPdfSlides[idx];
    currentSlideName = slide.name;

    showSection("processing");
    statusText.textContent = `Segmenting ${slide.name.replace(/_/g, " ")}...`;
    statusHint.textContent = `Slide ${idx + 1} of ${currentPdfSlides.length}`;

    try {
        const res = await fetch(`/segment-slide/${currentSessionId}/${idx}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ min_area: 100 }),
        });
        if (!res.ok) { const err = await res.json(); throw new Error(err.error || "Segmentation failed"); }
        const data = await res.json();
        allSegments = data.segments;
        selectedIndices.clear();
        segmentedSlides.add(slide.name);

        // Update the slide's segment count for when we go back
        currentPdfSlides[idx].segment_count = data.segment_count;

        renderResults();
    } catch (err) {
        showSection("slides");
        alert("Error: " + err.message);
    }
}

// ── Browse Runs ───────────────────────────────────────────────────────
async function loadRuns() {
    browseLoading.hidden = false;
    browseEmpty.hidden = true;
    runsList.hidden = true;

    try {
        const res = await fetch("/browse/runs");
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
        alert("Error loading runs: " + err.message);
    }
}

async function loadRun(runName) {
    currentRunName = runName;
    showSection("processing");
    statusText.textContent = "Loading run...";
    statusHint.textContent = "";

    try {
        const res = await fetch(`/browse/run/${runName}`);
        const data = await res.json();

        currentRunSlides = data.slides;
        browseImagesDir = data.images_dir;

        // Build slides data for the grid
        const slides = data.slides.map((s) => ({
            name: s.name,
            segment_count: s.segment_count,
            has_rendered: s.has_rendered,
        }));

        renderSlidesGrid(data.pdf_name.replace(/_/g, " "), slides, false);
    } catch (err) {
        showSection("browse");
        alert("Error loading run: " + err.message);
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
        const data = await res.json();
        allSegments = data.segments;
        selectedIndices.clear();
        renderResults();
    } catch (err) {
        showSection("slides");
        alert("Error: " + err.message);
    }
}

// ── Render Results ────────────────────────────────────────────────────
function renderResults() {
    showSection("results");
    segmentCount.textContent = `${allSegments.length} segments found`;

    // Show/hide slide navigation
    const hasSlides = currentMode === "pdf" || currentMode === "browse";
    backToSlidesBtn.hidden = !hasSlides;

    if (hasSlides && currentSlideIndex !== null) {
        slideNav.hidden = false;
        const totalSlides = currentMode === "browse" ? currentRunSlides.length : currentPdfSlides.length;
        slideNavLabel.textContent = `Slide ${currentSlideIndex + 1} / ${totalSlides}`;
        prevSlideBtn.disabled = currentSlideIndex <= 0;
        nextSlideBtn.disabled = currentSlideIndex >= totalSlides - 1;
    } else {
        slideNav.hidden = true;
    }

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
            toggleSelection(card, seg.index);
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
    const cards = [...segmentsGrid.querySelectorAll(".segment-card")];
    const segMap = new Map(allSegments.map((s) => [s.index, s]));
    cards.sort((a, b) => {
        const aSeg = segMap.get(parseInt(a.dataset.index));
        const bSeg = segMap.get(parseInt(b.dataset.index));
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
    pdfUploadSection.hidden = name !== "pdf-upload";
    browseSection.hidden = name !== "browse";
    slidesSection.hidden = name !== "slides";
    processingSection.hidden = name !== "processing";
    resultsSection.hidden = name !== "results";
}
