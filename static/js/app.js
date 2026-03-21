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
const downloadAllBtn = document.getElementById("download-all-btn");
const startOverBtn = document.getElementById("start-over-btn");
const areaFilter = document.getElementById("area-filter");
const areaValue = document.getElementById("area-value");

let currentSessionId = null;
let allSegments = [];

const downloadIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;

// Drag and drop
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

// Start over
startOverBtn.addEventListener("click", () => {
    showSection("upload");
    segmentsGrid.innerHTML = "";
    allSegments = [];
    currentSessionId = null;
    fileInput.value = "";
});

// Area filter
areaFilter.addEventListener("input", () => {
    const minArea = parseInt(areaFilter.value);
    areaValue.textContent = minArea.toLocaleString() + " px";
    filterSegments(minArea);
});

async function handleFile(file) {
    // Validate
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(jpe?g|png|webp|bmp|tiff?)$/i)) {
        alert("Unsupported file type. Please upload JPG, PNG, WebP, BMP, or TIFF.");
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        alert("File is too large. Maximum size is 50 MB.");
        return;
    }

    showSection("processing");
    statusText.textContent = "Uploading image...";
    statusHint.textContent = "";

    try {
        // Upload
        const formData = new FormData();
        formData.append("file", file);

        const uploadRes = await fetch("/upload", { method: "POST", body: formData });
        if (!uploadRes.ok) {
            const err = await uploadRes.json();
            throw new Error(err.error || "Upload failed");
        }
        const uploadData = await uploadRes.json();
        currentSessionId = uploadData.session_id;

        // Segment
        statusText.textContent = "Segmenting image...";
        statusHint.textContent = `${uploadData.width} × ${uploadData.height} px — this may take a few seconds`;

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

        renderResults();
    } catch (err) {
        showSection("upload");
        alert("Error: " + err.message);
    }
}

function renderResults() {
    showSection("results");
    segmentCount.textContent = `${allSegments.length} segments found`;

    // Set up area filter range
    if (allSegments.length > 0) {
        const maxArea = Math.max(...allSegments.map((s) => s.area));
        areaFilter.max = Math.floor(maxArea / 2);
        areaFilter.value = 0;
        areaValue.textContent = "0 px";
    }

    // Download all
    downloadAllBtn.onclick = () => {
        window.location.href = `/download-all/${currentSessionId}`;
    };

    // Render grid
    segmentsGrid.innerHTML = "";

    if (allSegments.length === 0) {
        segmentsGrid.innerHTML = `<div class="empty-state"><p>No segments found. Try uploading a different image.</p></div>`;
        return;
    }

    allSegments.forEach((seg) => {
        const card = document.createElement("div");
        card.className = "segment-card";
        card.dataset.area = seg.area;

        card.innerHTML = `
            <div class="segment-preview">
                <img src="/segment-image/${currentSessionId}/${seg.filename}"
                     alt="Segment ${seg.index}" loading="lazy">
            </div>
            <div class="segment-info">
                <div class="segment-meta">
                    ${seg.width} × ${seg.height} px<br>
                    ${seg.area.toLocaleString()} px area
                </div>
                <button class="segment-download" title="Download PNG">
                    ${downloadIcon}
                </button>
            </div>
        `;

        card.querySelector(".segment-download").addEventListener("click", () => {
            window.location.href = `/download/${currentSessionId}/${seg.filename}`;
        });

        segmentsGrid.appendChild(card);
    });

    updateVisibleCount();
}

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
    if (visible < total) {
        visibleCount.textContent = `(showing ${visible} of ${total})`;
    } else {
        visibleCount.textContent = "";
    }
}

function showSection(name) {
    uploadSection.hidden = name !== "upload";
    processingSection.hidden = name !== "processing";
    resultsSection.hidden = name !== "results";
}
