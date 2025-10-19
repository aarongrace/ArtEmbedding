console.log("starting labeler")
console.log("starting labeler")
console.log("starting labeler")

// save bandwidth
function saveBandwidth() {
    document.querySelectorAll("img").forEach(img => img.style.display = "none");
    // block related works section
    const relatedSection = document.querySelector('.wiki-layout-artworks-famous');
    if (relatedSection) {
        relatedSection.style.display = 'none';
    }

    const famousCarousels = document.querySelectorAll('.wiki-layout-artists-related.wiki-artwork-famous-carousel');
    famousCarousels.forEach(section => {
        section.style.display = 'none';
    });
}
saveBandwidth();

function main(){
    console.log("hello!")
    chrome.runtime.sendMessage({action: "SET_TITLE", title: getTitle()});
    // setTimeout(downloadVideo, 1000);
}


chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.action) {
    case "RUN_MAIN":
        main();
        break;

    case "downloadImage":
      sendResponse(downloadImage());
      break;

    case "processMovement":
      processMovement();
      break;
    
    case "getPaintingLinks":
        const hrefs = getPaintingLinks();
        sendResponse({ hrefs });
        break;

    default:
      console.warn("Unknown action:", message.action);
  }
});

// Example stub functions
function startTabWalking() {
    // TODO: scan the movement page for artist links and filter by nationality
    console.log("Walking through artist table...");
}

async function downloadImage() {
    console.log("Processing page once...");
    const res = await processPaintingDiv();
    console.log("Result:", res);
    if (!res) {
        console.log("Failed to process painting div.");
        return false;
    }
    return res;
}

function processMovement() {
    console.log("Processing movement page for artist links...");
    const allowedNationalities = [
        "German", "Dutch", "Flemish", "French", "Swiss", "Italian", "Spanish", "British", "English", 
        "Irish", "Austrian", "American", "Belgian", "Portuguese", "Scottish"
    ];

    // Select all artist list items
    const artistElements = document.querySelectorAll('div.masonry-text-view ul li');
    console.log(`Found ${artistElements.length} artist elements.`);

    const artists = Array.from(artistElements).map(li => {
        const anchor = li.querySelector('a');
        const name = anchor?.textContent.trim();
        const href = anchor?.href; // capture the artist page link
        const nationality = li.querySelector('span')?.textContent.replace(',', '').trim();

        return { name, href, nationality };
    }).filter(artist => allowedNationalities.includes(artist.nationality));
    console.log(`After filtering, ${artists.length} artists remain.`);

    const artistHrefs = artists.map(artist => artist.href);
    console.log(`Artist hrefs to process:`, artistHrefs);
    addTargetArtistsUrls(artistHrefs);
    // chrome.runtime.sendMessage({
    //     action: "PROCESS_MULTIPLE_ARTISTS",
    //     artistHrefs: artistHrefs
    // });
}


// for the all paintings of an artist page ////////////////////////////
function getPaintingLinks() {
  const links = [...document.querySelectorAll("ul.painting-list-text li a")];
  return links.map(a => a.href); // Absolute URLs
}