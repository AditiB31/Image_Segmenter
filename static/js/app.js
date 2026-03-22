const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadSection = document.getElementById("upload-section");
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

let currentSessionId = null;
let allSegments = [];
let selectedIndices = new Set();

const downloadIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;
const checkIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;
const zoomIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>`;

function getUpscale() {
    return parseInt(upscaleSelect.value);
}

// ── Drag and drop ────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── Start over ───────────────────────────────────────────────────────
startOverBtn.addEventListener("click", () => {
    showSection("upload");
    segmentsGrid.innerHTML = "";
    allSegments = [];
    selectedIndices.clear();
    currentSessionId = null;
    fileInput.value = "";
    previewImage.hidden = true;
    previewImage.src = "";
    sortBy.value = "area-desc";
    upscaleSelect.value = "2";
});

// ── Area filter ──────────────────────────────────────────────────────
areaFilter.addEventListener("input", () => {
    const minArea = parseInt(areaFilter.value);
    areaValue.textContent = minArea.toLocaleString() + " px";
    filterSegments(minArea);
});

// ── Sort handler ─────────────────────────────────────────────────────
sortBy.addEventListener("change", () => {
    sortSegments(sortBy.value);
});

// ── Keyboard shortcuts ───────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
    // Close modal on Escape
    if (e.key === "Escape") {
        if (previewModal.classList.contains("active")) {
            closeModal();
            return;
        }
        if (!resultsSection.hidden) {
            clearSelectionBtn.click();
        }
    }
    // Select all with Cmd/Ctrl+A
    if (!resultsSection.hidden && (e.ctrlKey || e.metaKey) && e.key === "a") {
        e.preventDefault();
        selectAllBtn.click();
    }
});

// ── Select all visible ───────────────────────────────────────────────
selectAllBtn.addEventListener("click", () => {
    const visibleCards = segmentsGrid.querySelectorAll('.segment-card:not([style*="display: none"])');
    visibleCards.forEach((card) => {
        const idx = parseInt(card.dataset.index);
        selectedIndices.add(idx);
        card.classList.add("selected");
    });
    updateSelectionUI();
});

// ── Clear selection ──────────────────────────────────────────────────
clearSelectionBtn.addEventListener("click", () => {
    selectedIndices.clear();
    segmentsGrid.querySelectorAll(".segment-card.selected").forEach((card) => {
        card.classList.remove("selected");
    });
    updateSelectionUI();
});

// ── Download ─────────────────────────────────────────────────────────
downloadBtn.addEventListener("click", async () => {
    const indices = selectedIndices.size > 0
        ? [...selectedIndices]
        : allSegments.map((s) => s.index);

    const upscale = getUpscale();
    downloadBtn.disabled = true;
    downloadBtn.textContent = "Preparing…";

    try {
        const res = await fetch(`/download-all/${currentSessionId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ indices, upscale }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Download failed");
        }

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

// ── Modal ────────────────────────────────────────────────────────────
function openModal(seg) {
    modalImage.src = `/segment-image/${currentSessionId}/${seg.filename}`;
    modalInfo.innerHTML = `
        <strong>Segment ${seg.index}</strong><br>
        ${seg.width} × ${seg.height} px &middot;
        ${seg.area.toLocaleString()} px² &middot;
        IoU ${seg.predicted_iou}
    `;
    modalDownload.onclick = () => {
        window.location.href = `/download/${currentSessionId}/${seg.filename}?upscale=${getUpscale()}`;
    };
    previewModal.classList.add("active");
}

function closeModal() {
    previewModal.classList.remove("active");
}

modalClose.addEventListener("click", closeModal);
previewModal.addEventListener("click", (e) => {
    if (e.target === previewModal) closeModal();
});

// ── File handling ────────────────────────────────────────────────────
async function handleFile(file) {
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpe?g|png|webp|bmp|tiff?)$/i)) {
        alert("Unsupported file type. Please upload JPG, PNG, WebP, BMP, or TIFF.");
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        alert("File is too large. Maximum size is 50 MB.");
        return;
    }

    // Show image preview immediately via object URL
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
        if (!uploadRes.ok) {
            const err = await uploadRes.json();
            throw new Error(err.error || "Upload failed");
        }
        const uploadData = await uploadRes.json();
        currentSessionId = uploadData.session_id;

        statusText.textContent = "Segmenting image...";
        statusHint.textContent = `${uploadData.width} × ${uploadData.height} px — this may take a moment`;

        const segRes = await fetch(`/segment/${currentSessionId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ min_area: 100 }),
        });
        if (!segRes.ok) {
            const err = await segRes.json();
            throw new Error(err.error || "Segmentation failed");
        }
        const segData = await segRes.json();
        allSegments = segData.segments;
        selectedIndices.clear();

        renderResults();
    } catch (err) {
        showSection("upload");
        alert("Error: " + err.message);
    }
}

// ── Render results ───────────────────────────────────────────────────
function renderResults() {
    showSection("results");
    segmentCount.textContent = `${allSegments.length} segments found`;

    if (allSegments.length > 0) {
        const maxArea = Math.max(...allSegments.map((s) => s.area));
        areaFilter.max = Math.floor(maxArea / 2);
        areaFilter.value = 0;
        areaValue.textContent = "0 px";
    }

    segmentsGrid.innerHTML = "";

    if (allSegments.length === 0) {
        segmentsGrid.innerHTML = `<div class="empty-state"><p>No segments found. Try uploading a different image.</p></div>`;
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
                <img src="/segment-image/${currentSessionId}/${seg.filename}"
                     alt="Segment ${seg.index}" loading="lazy">
            </div>
            <div class="segment-info">
                <div class="segment-meta">
                    ${seg.width} × ${seg.height} px<br>
                    ${seg.area.toLocaleString()} px area
                </div>
                <div class="segment-actions">
                    <button class="segment-zoom" title="Preview">
                        ${zoomIcon}
                    </button>
                    <button class="segment-download" title="Download full-res PNG">
                        ${downloadIcon}
                    </button>
                </div>
            </div>
        `;

        // Toggle selection on card click (not on action buttons)
        card.addEventListener("click", (e) => {
            if (e.target.closest(".segment-download") || e.target.closest(".segment-zoom")) return;
            toggleSelection(card, seg.index);
        });

        // Zoom preview
        card.querySelector(".segment-zoom").addEventListener("click", (e) => {
            e.stopPropagation();
            openModal(seg);
        });

        // Individual download (uses selected upscale level)
        card.querySelector(".segment-download").addEventListener("click", (e) => {
            e.stopPropagation();
            window.location.href = `/download/${currentSessionId}/${seg.filename}?upscale=${getUpscale()}`;
        });

        segmentsGrid.appendChild(card);
    });

    updateSelectionUI();
    updateVisibleCount();
}

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
        downloadBtn.textContent = `Download All ${upscale}× ZIP`;
    } else {
        selectedCount.textContent = `${n} selected`;
        downloadBtn.textContent = `Download ${n} Selected ${upscale}× ZIP`;
    }
    downloadBtn.disabled = allSegments.length === 0;
}

// Update button text when upscale changes
upscaleSelect.addEventListener("change", updateSelectionUI);

function filterSegments(minArea) {
    const cards = segmentsGrid.querySelectorAll(".segment-card");
    cards.forEach((card) => {
        const area = parseInt(card.dataset.area);
        card.style.display = area >= minArea ? "" : "none";
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

function showSection(name) {
    uploadSection.hidden = name !== "upload";
    processingSection.hidden = name !== "processing";
    resultsSection.hidden = name !== "results";
}
