// Helper to send messages to content scripts
function sendMessageToContent(action) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0].id) {
      chrome.tabs.sendMessage(tabs[0].id, { action });
    }
  });
}

document.getElementById("downloadImage").addEventListener("click", () => {
  console.log("========== downloadImage ==========");
  sendMessageToContent("downloadImage");
});


document.getElementById("showData").addEventListener("click", () => {
  console.log("========== Showing stored data ==========");
  showStoredData();
});

document.getElementById("downloadData").addEventListener("click", () => {
  console.log("========== Downloading stored data as JSON ==========");
  downloadMetadataAsJson();
});
document.getElementById("downloadPaintingData").addEventListener("click", () => {
  console.log("========== Downloading paintings metadata as JSON ==========");
  downloadPaintingsMetadataAsJson();
});

document.getElementById("deleteAll").addEventListener("click", () => {
  console.log("========== Deleting all stored data ==========");
  deleteAllStoredData();
});

document.getElementById("addTest").addEventListener("click", () => {
  console.log("========== Adding test painting ==========");
  addTestPainting();
});


document.getElementById("processArtist").addEventListener("click", () => {
  console.log("========== Processing artist ==========");

  // Example hard-coded artist URL (can make this dynamic later)
  const artistBaseUrl = "https://www.wikiart.org/en/henry-fuseli";

  chrome.runtime.sendMessage({
    action: "PROCESS_ARTIST",
    artistUrl: artistBaseUrl,
  });
});

document.getElementById('processMovementPage').addEventListener('click', async () => {
  console.log("========== Processing movement page ==========");
  sendMessageToContent("processMovement");
});


document.getElementById('startDownloadLoop').addEventListener('click', async () => {
  console.log("========== Starting download loop ==========");
  chrome.runtime.sendMessage({ action: "START_DOWNLOAD_LOOP" });
});

document.getElementById('stopDownloadLoop').addEventListener('click', async () => {
  console.log("========== Stopping download loop ==========");
  chrome.runtime.sendMessage({ action: "STOP_DOWNLOAD_LOOP" });
});

async function updateStatus() {
  const statusDiv = document.getElementById('status');
  const progressStats = await getProgressStats();
  const paintingsCount = progressStats.paintingsCount || 0;
  const processedUrlsCount = progressStats.processedUrlsCount || 0;
  const processedArtistUrlsCount = progressStats.processedArtistUrlsCount || 0;
  const targetArtistUrlsCount = progressStats.targetArtistUrlsCount || 0;
  statusDiv.innerHTML = `
    <p>Paintings stored: ${paintingsCount}</p>
    <p>Processed URLs: ${processedUrlsCount}</p>
    <p>Processed Artists: ${processedArtistUrlsCount} / ${targetArtistUrlsCount}</p>
  `;
}

document.getElementById("upload-btn").addEventListener("click", () => {
    const fileInput = document.getElementById("upload-json");
    const file = fileInput.files[0];
    uploadMetadataFromJson(file);
});
updateStatus();