// Function to group ICON tabs
async function groupIconTabs() {
    const tabs = await chrome.tabs.query({
        url: [
            "https://uiowa.instructure.com/*",
        ],
    });

    const collator = new Intl.Collator();
    tabs.sort((a, b) => collator.compare(a.title, b.title));

    const template = document.getElementById("li_template");
    const elements = new Set();
    for (const tab of tabs) {
        const element = template.content.firstElementChild.cloneNode(true);
        const title = tab.title ? tab.title.split("-")[0].trim() : "No Title";

        let pathname = tab.url ? tab.url : "No URL";
        if (pathname.includes("://")) {
            pathname = pathname.split("://")[1];
        }

        element.querySelector(".title").textContent = title;
        element.querySelector(".pathname").textContent = pathname;
        element.querySelector("a").addEventListener("click", async () => {
            await chrome.tabs.update(tab.id, { active: true });
            await chrome.windows.update(tab.windowId, { focused: true });
        });

        elements.add(element);
    }
    document.querySelector("ul").append(...elements);

    const tabIds = tabs.map(({ id }) => id);
    const group = await chrome.tabs.group({ tabIds });
    await chrome.tabGroups.update(group, { title: "ICON" });
}

// Function to group all tabs by their host
async function groupAllTabsByHost() {
    const tabs = await chrome.tabs.query({});

    const collator = new Intl.Collator();
    tabs.sort((a, b) => collator.compare(a.title, b.title));

    const template = document.getElementById("li_template");
    const elements = new Set();
    const hostGroups = {};

    for (const tab of tabs) {
        const element = template.content.firstElementChild.cloneNode(true);
        const title = tab.title ? tab.title.split("-")[0].trim() : "No Title";

        let pathname = tab.url ? tab.url : "No URL";
        if (pathname.includes("://")) {
            pathname = pathname.split("://")[1];
        }

        let host = "No Host";
        try {
            const url = new URL(tab.url);
            host = url.host;
        } catch (e) {
            console.error("Invalid URL:", tab.url);
        }

        if (!hostGroups[host]) {
            hostGroups[host] = [];
        }
        hostGroups[host].push(tab.id);

        element.querySelector(".title").textContent = title;
        element.querySelector(".pathname").textContent = pathname;
        element.querySelector("a").addEventListener("click", async () => {
            await chrome.tabs.update(tab.id, { active: true });
            await chrome.windows.update(tab.windowId, { focused: true });
        });

        elements.add(element);
    }
    document.querySelector("ul").append(...elements);

    for (const host in hostGroups) {
        const group = await chrome.tabs.group({ tabIds: hostGroups[host] });
        await chrome.tabGroups.update(group, { title: host });
    }
}

// Add event listeners to buttons
document.getElementById("group_icon_tabs").addEventListener("click", groupIconTabs);
document.getElementById("group_all_tabs").addEventListener("click", groupAllTabsByHost);

