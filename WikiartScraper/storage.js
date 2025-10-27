let nextIdCache = null;

// Initialize the ID pool (IDs 0–30000 as strings, minus already used IDs)
async function initIdPool() {
    const stored = await new Promise((resolve, reject) => {
        chrome.storage.local.get({ paintings: [] }, (result) => {
            if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
            resolve(result.paintings);
        });
    });

    // Collect used IDs as strings
    const usedIds = new Set(stored.map(p => p.id).filter(Boolean));

    const availableIds = [];
    // IDs <= 1374
    for (let i = 0; i <= 1374; i++) {
        const idStr = String(i).padStart(6, "0");
        if (!usedIds.has(idStr)) {
            availableIds.push(idStr);
        }
    }

    // IDs > 30928
    for (let i = 30929; i <= 30000 + 10000; i++) { // choose upper limit as needed
        const idStr = String(i).padStart(6, "0");
        if (!usedIds.has(idStr)) {
            availableIds.push(idStr);
        }
    }

    nextIdCache = availableIds;
    console.log(`========== ID pool initialized with ${availableIds.length} available IDs ==========`);
    console.log(nextIdCache);
}
if (!nextIdCache) {
    initIdPool();
}


// Pop the next available string ID
async function getNextId() {
    if (!nextIdCache || nextIdCache.length === 0) {
        throw new Error("ID pool is not initialized or empty. Call initIdPool() first.");
    }

    return nextIdCache.shift();
}

// Async function to save painting with unique ID
async function savePainting(painting) {
    try {
        // Only generate a new ID if painting.id does not already exist
        if (!painting.id) {
            const id = await getNextId();
            painting.id = id;
        }

        // Get current paintings, append, and save
        const stored = await new Promise((resolve, reject) => {
            chrome.storage.local.get({ paintings: [] }, (result) => {
                if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
                resolve(result.paintings);
            });
        });

        stored.push(painting);

        await new Promise((resolve, reject) => {
            chrome.storage.local.set({ paintings: stored }, () => {
                if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
                resolve();
            });
        });

        console.log(`========== Saving painting with ID ${painting.id} | Title: ${painting.title} ==========`);
        return painting;

    } catch (err) {
        console.error("Error saving painting:", err);
        throw err;
    }
}


function addProcecssedUrl(url) {
    chrome.storage.local.get({ processedUrls: [] }, (result) => {
        const processedUrls = result.processedUrls;
        if (!processedUrls.includes(url)) {
            processedUrls.push(url);
            chrome.storage.local.set({ processedUrls }, () => {
                if (chrome.runtime.lastError) {
                    console.error("Error saving processed URL:", chrome.runtime.lastError);
                } else {
                    console.log(`========== Added processed URL: ${url} ==========`);
                }
            });
        }
    });
}

function isUrlProcessed(url) {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get({ processedUrls: [] }, (result) => {
            if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
            const processedUrls = result.processedUrls;
            resolve(processedUrls.includes(url));
        });
    }
    );
}

function getAllProcessedUrls() {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get({ processedUrls: [] }, (result) => {
            if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
            resolve(result.processedUrls);
        });
    });
}

function addProcessedArtistUrl(url) {
    chrome.storage.local.get({ processedArtistUrls: [] }, (result) => {
        const processedArtistUrls = result.processedArtistUrls;
        if (!processedArtistUrls.includes(url)) {
            processedArtistUrls.push(url);
            chrome.storage.local.set({ processedArtistUrls }, () => {
                if (chrome.runtime.lastError) {
                    console.error("Error saving processed artist URL:", chrome.runtime.lastError);
                } else {
                    console.log(`========== Added processed artist URL: ${url} ==========`);
                }
            });
        }
    });
}

function getAllProcessedArtistUrls() {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get({ processedArtistUrls: [] }, (result) => {
            if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
            resolve(result.processedArtistUrls);
        });
    });
}


function addTargetArtistsUrls(urls) {
    chrome.storage.local.get({ targetArtistUrls: [] }, (result) => {
        const targetArtistUrls = result.targetArtistUrls;
        const newUrls = urls.filter(url => !targetArtistUrls.includes(url));
        if (newUrls.length > 0) {
            const updatedUrls = targetArtistUrls.concat(newUrls);
            chrome.storage.local.set({ targetArtistUrls: updatedUrls }, () => {
                if (chrome.runtime.lastError) {
                    console.error("Error saving target artist URLs:", chrome.runtime.lastError);
                } else {
                    console.log(`========== Added ${newUrls.length} target artist URLs ==========`);
                }
            });
        } else {
            console.log("No new target artist URLs to add.");
        }
    });
}

function getAllTargetArtistUrls() {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get({ targetArtistUrls: [] }, (result) => {
            if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
            resolve(result.targetArtistUrls);
        });
    });
}



// Show all stored paintings
function showStoredData() {
    chrome.storage.local.get({ paintings: [] }, (result) => {
        console.log("========== Stored Paintings ==========");
        console.log(result.paintings);
    });
    chrome.storage.local.get({ processedUrls: [] }, (result) => {
        console.log("========== Processed URLs ==========");
        console.log(result.processedUrls);
    });
    chrome.storage.local.get({ processedArtistUrls: [] }, (result) => {
        console.log("========== Processed Artist URLs ==========");
        console.log(result.processedArtistUrls);
    });
    chrome.storage.local.get({ targetArtistUrls: [] }, (result) => {
        console.log("========== Target Artist URLs ==========");
        console.log(result.targetArtistUrls);
    });
}

// Delete all stored data
function deleteAllStoredData() {
    chrome.storage.local.set({ paintings: [] }, () => {
        console.log("========== All stored data deleted ==========");
    });
    chrome.storage.local.set({ processedUrls: [] }, () => {
        console.log("========== All processed URLs deleted ==========");
    });
    chrome.storage.local.set({ processedArtistUrls: [] }, () => {
        console.log("========== All processed artist URLs deleted ==========");
    }
    );
    chrome.storage.local.set({ targetArtistUrls: [] }, () => {
        console.log("========== All target artist URLs deleted ==========");
    });
}

// Add a test painting (for debug)
function addTestPainting() {
    ({
        title: "Test Painting",
        artist: "Test Artist",
        style: "Impressionism",
        genre: "Landscape",
        imageUrl: "http://example.com/image.jpg"
    });
}

function downloadMetadataAsJson() {
    chrome.storage.local.get({ paintings: [], processedUrls: [], processedArtistUrls: [], targetArtistUrls: [] }, (result) => {
        const paintings = result.paintings;
        const processedUrls = result.processedUrls || [];
        const processedArtistUrls = result.processedArtistUrls || [];
        const targetArtistUrls = result.targetArtistUrls || [];
        const data = {
            paintings,
            processedUrls,
            processedArtistUrls,
            targetArtistUrls
        };
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "raw_wikiart_metadata.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log("========== Metadata downloaded as paintings_metadata.json ==========");
    });
}

function downloadPaintingsMetadataAsJson() {
    chrome.storage.local.get({ paintings: [] }, (result) => {
        const paintings = result.paintings;
        const data = {};
        paintings.forEach(p => {
            data[p.id] = p;
        });
        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "paintings_metadata.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log("========== Paintings metadata downloaded as paintings_metadata.json ==========");
    });
}


function uploadMetadataFromJson(file) {
    if (!file) {
        console.error("No file provided for upload.");
        return;
    }

    const reader = new FileReader();

    reader.onload = (event) => {
        try {
            const json = event.target.result;
            const data = JSON.parse(json);

            // Ensure all keys exist and are arrays
            const paintings = Array.isArray(data.paintings) ? data.paintings : [];
            const processedUrls = Array.isArray(data.processedUrls) ? data.processedUrls : [];
            const processedArtistUrls = Array.isArray(data.processedArtistUrls) ? data.processedArtistUrls : [];
            const targetArtistUrls = Array.isArray(data.targetArtistUrls) ? data.targetArtistUrls : [];

            chrome.storage.local.set({
                paintings,
                processedUrls,
                processedArtistUrls,
                targetArtistUrls
            }, () => {
                if (chrome.runtime.lastError) {
                    console.error("Error uploading metadata:", chrome.runtime.lastError);
                } else {
                    console.log("========== Local storage successfully overwritten with uploaded metadata ==========");
                    showStoredData(); // optional: print to console to verify
                }
            });
        } catch (err) {
            console.error("Error parsing uploaded JSON:", err);
        }
    };

    reader.onerror = (err) => {
        console.error("Error reading file:", err);
    };

    reader.readAsText(file);
}


function getProgressStats() {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get({ paintings: [], processedUrls: [], processedArtistUrls: [], targetArtistUrls: [] }, (result) => {
            if (chrome.runtime.lastError) return reject(chrome.runtime.lastError);
            const paintingsCount = result.paintings.length;
            const processedUrlsCount = result.processedUrls.length;
            const processedArtistUrlsCount = result.processedArtistUrls.length;
            const targetArtistUrlsCount = result.targetArtistUrls.length;
            resolve({
                paintingsCount,
                processedUrlsCount,
                processedArtistUrlsCount,
                targetArtistUrlsCount
            });
        }
        );
    });
}

// (async () => {
//   // ✅ STEP 1: Define artists to remove (most distinctive last-name keyword)
//   const artistsToRemove = [
//     "honthorst",
//     "guercino",
//     "ribera",
//     "vouet",
//     "tournier",
//     "fiasella",
//     "venne",
//     "seghers",
//     "deruet",
//     "terbrugghen",
//     "velde",
//     "avercamp",
//     "vos" // careful: Cornelis de Vos
//   ];

//   // ✅ STEP 2: Get processedArtists from storage
//   let processedArtistsUrls = await getAllProcessedArtistUrls();


//   console.log(`Before cleanup: ${processedArtistsUrls.length} artists stored.`);

//   // ✅ STEP 3: Filter out matching artists
//   processedArtistsUrls = processedArtistsUrls.filter(urlOrName => {
//     const lower = urlOrName.toLowerCase();
//     return !artistsToRemove.some(artist => lower.includes(artist));
//   });

//   console.log(`After cleanup: ${processedArtistsUrls.length} artists remain.`);
//   chrome.storage.local.set({ processedArtistUrls: processedArtistsUrls }, () => {
//       if (chrome.runtime.lastError) {
//           console.error("Error updating processed artist URLs:", chrome.runtime.lastError);
//       } else {
//           console.log("Processed artist URLs updated successfully.");
//       }
//     });

// })();
