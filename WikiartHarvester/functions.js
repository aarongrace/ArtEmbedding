async function processPaintingDiv() {
    const paintingData = getPaintingData();
    if (!paintingData) {
        return null;
    }

    return await getImageFromPage(paintingData);
}

function getPaintingData() {
    const container = document.querySelector('.wiki-layout-artist-info.wiki-layout-artwork-info');
    if (!container) return null;

    // --- Helper functions ---
    function getLiByLabel(labelText) {
        const lis = container.querySelectorAll('ul li.dictionary-values, ul li.dictionary-values-gallery, ul li');
        for (const li of lis) {
            const label = li.querySelector('s');
            if (label && label.textContent.trim().toLowerCase() === labelText.toLowerCase()) {
                return li;
            }
        }
        return null;
    }

    function getLiValuesArray(li) {
        if (!li) return [];
        const links = li.querySelectorAll('a');
        const textValues = Array.from(links)
            .map(el => el.textContent.trim())
            .filter(Boolean);

        // Fallback: plain text split by commas
        if (!textValues.length && li) {
            const text = li.textContent.replace(/.*?:/, '').trim();
            if (text) return text.split(',').map(s => s.trim()).filter(Boolean);
        }

        return textValues;
    }

    // --- Extract painting info ---
    const title = container.querySelector('header h1')?.textContent.trim() || '';
    const artist = container.querySelector('header h2 a')?.textContent.trim() || '';
    const year = getLiByLabel('Date:')?.querySelector('[itemprop="dateCreated"]')?.textContent.trim() || '';

    const styles = getLiValuesArray(getLiByLabel('Style:'));
    const genres = getLiValuesArray(getLiByLabel('Genre:'));
    const media = getLiValuesArray(getLiByLabel('Media:'));

    const location = getLiByLabel('Location:')?.querySelector('span')?.textContent.trim() || '';
    const dimensions = getLiByLabel('Dimensions:')?.textContent.replace('Dimensions:', '').trim() || '';
    const maxResolution = container.querySelector('.max-resolution')?.textContent.trim() || '';

    const tags = Array.from(container.querySelectorAll('.tags-cheaps__item__ref'))
        .map(el => el.textContent.trim())
        .filter(Boolean);

    // --- FILTERING ---
    const allowedGenres = [
        "battle",
        "history",
        "genre",
        "religious painting",
        "landscape",
        "portrait",
        "mythological",
        "literary",
    ];
    const disallowedGenres = [
      "scultpure",
      "sketch",
      "study",
      "drawing"
    ];
    const allowedMovements = [
        "baroque",
        "rococo",
        "neoclassicism",
        "romanticism",
        "orientalism",
        "realism",
        "impressionism"
    ];
    const disallowedMedia = [
        "sketch",
        "drawing",
        "print",
        "etching",
        "statue",
        "sculpture",
        "lithograph",
        "woodcut",
        "engraving",
        "ceramic",
        "marble",
        "chalk",
        "ink",
        "pencil"
    ];
    const hasDisallowedMedia = media.some(m =>
        disallowedMedia.some(disallowed => m.toLowerCase().includes(disallowed))
    );
    if (hasDisallowedMedia) {
        console.log(`========== Skipping painting: found disallowed media in [${media.join(', ')}] ==========`);
        return null;
    }

    const matchedGenre = genres.find(g =>
        allowedGenres.some(allowed => g.toLowerCase().includes(allowed))
    );
    if (!matchedGenre) {
        console.log(`========== Skipping painting: no allowed genre found in [${genres.join(', ')}] ==========`);
        return null;
    }
    const hasDisallowedGenre = genres.some(g =>
        disallowedGenres.some(disallowed => g.toLowerCase().includes(disallowed))
    );
    if (hasDisallowedGenre) {
        console.log(`========== Skipping painting: found disallowed genre in [${genres.join(', ')}] ==========`);
        return null;
    }

    const matchedMovement = styles.find(s => allowedMovements.includes(s.toLowerCase()));
    if (!matchedMovement) {
        console.log(`========== Skipping painting: no allowed movement found in [${styles.join(', ')}] ==========`);
        return null;
    }

    const disallowedTags = [
        "sketch",
        "stock photography",
        "carving",
        "relief",
        "statue",
        "sculpture",
    ]
    const hasDisallowedTag = tags.some(t =>
        disallowedTags.some(disallowed => t.toLowerCase().includes(disallowed))
    );
    if (hasDisallowedTag) {
        console.log(`========== Skipping painting: found disallowed tag in [${tags.join(', ')}] ==========`);
        return null;
    }

    const url = window.location.href;



    // --- Build result object ---
    const paintingData = {
        title,
        artist,
        year,
        styles,
        genres,
        media,
        location,
        dimensions,
        maxResolution,
        tags,
        url
    };

    console.log(`========== Accepted painting: ${title} (${matchedMovement}, ${matchedGenre}) ==========`);
    console.log(paintingData);
    return paintingData;

}


function sanitizeFilename(str) {
    const replacedSlash = str.replace(/\//g, "%");
    console.log(`Sanitized slashes: ${replacedSlash}`);
    const replacedForbidden = replacedSlash.replace(/[\\:*?"<>|]/g, "_");
    console.log(`Sanitized forbidden chars: ${replacedForbidden}`);
    const lowercased = replacedForbidden.toLowerCase();
    console.log(`Lowercased: ${lowercased}`);

    return str
        .replace(/\//g,"—")
        .replace(/[\\:*?"<>|]/g, "_")  // only replace truly forbidden chars
        .toLowerCase();
}

function buildFilename(painting) {
    // Zero pad ID to 6 digits
    const paddedId = String(painting.id).padStart(6, "0");

    // // Get artist last name (if available)
    // const lastName = painting.artist.includes(" ")
    //     ? painting.artist.split(" ").slice(-1)[0]
    //     : painting.artist;

    // // Shorten movement + genre(s)
    // const shortMovement = painting.styles?.[0] ? painting.styles[0].slice(0, 4) : "unkn";
    // const shortGenres = painting.genres?.length
    //     ? painting.genres.map(g => g.slice(0, 4)).join("-")
    //     : "none";

    const urlEnding = painting.url.split("/en/")[1] || "unknown";
    return sanitizeFilename(`${paddedId}_${urlEnding}`);


    // Build filename
    // return sanitizeFilename( `${paddedId}_${lastName}-${painting.title}-${shortMovement}-${shortGenres}-${urlEnding}` );
}

async function getImageFromPage(paintingData) {
    const link = document.querySelector('a.all-sizes');
    if (!link) {
        console.log("========== 'View all sizes' link not found ==========");
        return false;
    }
    console.log("========== Opening 'View all sizes' ==========");
    link.click();

    const delay = Math.random() * (1100 - 500) + 500;
    console.log(`========== Waiting ${Math.round(delay)}ms before downloading ==========`);
    await new Promise(resolve => setTimeout(resolve, delay));


    // find the largest resolution link
    const items = document.querySelectorAll('.view-thumnails-sizes-item .thumbnail-item');
    if (!items.length) {
        console.log("========== No thumbnail items found ==========");
        return false;
    }

    let largest = null;
    let maxArea = 0;

    items.forEach(item => {
        const link = item.querySelector('a');
        if (!link) return;

        const resolutionText = link.textContent.trim();
        const match = resolutionText.match(/(\d+)\s*x\s*(\d+)/);
        if (!match) return;

        const width = parseInt(match[1], 10);
        const height = parseInt(match[2], 10);
        const area = width * height;

        if (area > maxArea) {
            maxArea = area;
            largest = link;
        }
    });

    if (largest) {
        const imageUrl = largest.href;

        const id = await getNextId();
        paintingData.id = id;
        const filename = buildFilename(paintingData);
        console.log(`Built filename: ${filename}`);
        console.log(`========== Downloading largest image: ${largest.textContent} | Filename: ${filename} ==========`);

        chrome.runtime.sendMessage({
            action: "DOWNLOAD_IMAGE",
            url: imageUrl,
            filename: filename
        });
        res = await savePainting(paintingData);
        return res;
    } else {
        console.log("========== No valid resolution found ==========");
        return false;
    }
    return true;
}



// accepted countries: belgian, french, german, italian, dutch, spanish, english, british