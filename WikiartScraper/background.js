// the stop loop function doesn't work
importScripts('storage.js');
console.log("background.js loaded");

let isDownloadLoopActive = false;

let backgroundWindow = null; // global ref
async function ensureBackgroundWindow() {
  if (backgroundWindow) {
    try {
      // Check if window still exists
      await chrome.windows.get(backgroundWindow.id);
      return backgroundWindow;
    } catch (e) {
      console.log("Background window no longer exists, recreating...");
      backgroundWindow = null;
    }
  }

  if (!backgroundWindow) {
    // Create a new background window (minimized so user doesn’t see it)
    backgroundWindow = await chrome.windows.create({
      focused: false,
      state: "minimized"
    });

    console.log("Created new background window:", backgroundWindow.id);
    return backgroundWindow;
  }
}


async function safeCreateTab(tabOptions, retries = 20, delay = 100) {
  for (let i = 0; i < retries; i++) {
    try {
      return await chrome.tabs.create(tabOptions);
    } catch (err) {
      if (err.message.includes("Tabs cannot be edited right now")) {
        console.warn("Retrying tab creation...", i + 1);
        await new Promise(res => setTimeout(res, delay));
      } else {
        throw err;
      }
    }
  }
  throw new Error("Failed to create tab after multiple retries");
}

function getRandomDelay(base = 15000, jitter = 300) {
  // Random delay: between (base - jitter) and (base + jitter)
  const min = Math.max(1000, base - jitter);
  const max = base + jitter;
  return Math.floor(Math.random() * (max - min + 1) + min);
}


chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg.action === "PROCESS_ARTIST") {
      processArtist(msg.artistUrl);
  }

  if (msg.action === "START_DOWNLOAD_LOOP") {
      startDownloadLoop();
  }
  if (msg.action === "STOP_DOWNLOAD_LOOP") {
      stopDownloadLoop();
  }




  if (msg.action === "DOWNLOAD_IMAGE") {
    console.log(`========== Background: Starting download for ${msg.filename} ==========`);

    chrome.downloads.download({
      url: msg.url,
      filename: `wikiart/${msg.filename}.jpg`, // saves inside a "wikiart" folder
      conflictAction: "uniquify", // if file exists, Chrome will append a number
      saveAs: false // no prompt, download silently
    }, (downloadId) => {
      if (chrome.runtime.lastError) {
        console.error("========== Download error ==========", chrome.runtime.lastError);
      } else {
        console.log(`========== Download started | ID: ${downloadId} ==========`);
      }
    });
  }
});


async function processArtist(artistUrl) {
  console.log(`Processing artist: ${artistUrl}`);

  // Convert base artist URL to text-list page
  const artistListUrl = `${artistUrl}/all-works/text-list`;

  const listTab = await chrome.tabs.create({ url: artistListUrl, active: false });
  console.log(`Opened tab for artist: ${artistListUrl} | Tab ID: ${listTab.id}`);

  await new Promise(resolve => {
    chrome.tabs.onUpdated.addListener(function listener(tabId, info) {
      if (tabId === listTab.id && info.status === "complete") {
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    });
  });

  const response = await chrome.tabs.sendMessage(listTab.id, {
    action: "getPaintingLinks"
  });

  let artworkHrefs =  response.hrefs; // Array of URLs
  console.log(`Found ${artworkHrefs.length} artworks for artist.`);
  console.log(artworkHrefs);
  chrome.tabs.remove(listTab.id);

  const processedUrls = await getAllProcessedUrls();
  artworkHrefs = artworkHrefs.filter(href => !processedUrls.includes(href));
  console.log(`After filtering already processed, ${artworkHrefs.length} artworks remain.`);

  while (artworkHrefs.length > 0) {
    if (!isDownloadLoopActive) {
      console.log("Download loop has been stopped.");
      return;
    }
    const nextUrl = artworkHrefs.shift();

    console.log(`Navigating to artwork: ${nextUrl}`);
    const bgWindow = await ensureBackgroundWindow();
    const artTab = await safeCreateTab({ windowId: bgWindow.id, url: nextUrl, active: false });

    await new Promise(resolve => {
      chrome.tabs.onUpdated.addListener(function listener(tabId, info) {
        if (tabId === artTab.id && info.status === "complete") {
          chrome.tabs.onUpdated.removeListener(listener);
          resolve();
        }
      });
    });

    const processed = await chrome.tabs.sendMessage(artTab.id, { action: "downloadImage" });
    if (!processed) {
      console.log(`Did not download ${nextUrl}, likely it just doesn't have the right genre/movement`);
    } else {
      console.log("Artwork downloaded successfully.");
    }
    const timeOutTime = getRandomDelay(4000, 1000);
    await new Promise(r => setTimeout(r, timeOutTime)); // short delay before next artwork


    chrome.tabs.remove(artTab.id);
    addProcecssedUrl(nextUrl);
  }
  console.log(`Finished processing artist: ${artistUrl}`);
  addProcessedArtistUrl(artistUrl);
  return true;
}


async function startDownloadLoop(){
  isDownloadLoopActive = true;
  const targetArtists = await getAllTargetArtistUrls();
  console.log(`Total target artists: ${targetArtists.length}`);
  const processedArtists = await getAllProcessedArtistUrls();
  console.log(`Total processed artists: ${processedArtists.length}`);
  const unprocessedArtists = targetArtists.filter(artist => !processedArtists.includes(artist));
  console.log(`After filtering processed, ${unprocessedArtists.length} artists remain.`);
  console.log(unprocessedArtists);

  while (unprocessedArtists.length > 0 && isDownloadLoopActive) {
    const artist = unprocessedArtists.shift();
    console.log(`Next artist: ${artist}`);
    await processArtist(artist);

  }
}

function stopDownloadLoop() {
  isDownloadLoopActive = false;
  console.log("Download loop stopped.");
}