async function goToNextPainting() {
    console.log("Looping to next painting...");

    const nextBtn = document.querySelector('a.wiki-breadcrumbs-btns-next');
    if (!nextBtn) {
        console.log("Next painting button not found.");
        return false;
    }

    console.log("Clicking next painting button...");
    nextBtn.click();
        
    const randomDelay = Math.floor(Math.random() * (500)) + 300;
    await new Promise(r => setTimeout(r, randomDelay)); // wait for page to load
    return true;
}