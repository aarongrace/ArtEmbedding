// Function to retrieve assignments_info from chrome.storage.local
async function getAssignmentsInfo() {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get("assignments_info", function (data) {
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError);
            } else {
                resolve(data.assignments_info);
            }
        });
    });
}

// Function to save assignments_info to chrome.storage.local
async function setAssignmentsInfo() {
    console.log("saving assignments_info to storage", assignments_info);

    return new Promise((resolve, reject) => {
        chrome.storage.local.set({ "assignments_info": assignments_info }, function () {
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError);
            } else {
                resolve();
                console.log("assignments_info saved to storage", assignments_info);
            }
        });
    });
}

// Function to find assignment by ID
function findAssignmentById(id) {
    for (const [key, value] of Object.entries(assignments_info)) {
        if (value.ids.includes(id)) {
            return [key, value];
        }
    }
    return [null, null];
}

// Function to replace the designate button with the designate div
function replaceWithDesignateDiv() {
    designateDiv.style.display = "block";
    designateButton.style.display = "none";
    console.log(assignments_info);
    Object.keys(assignments_info).forEach((aName) => {
        const option = document.createElement('option');
        option.value = aName;
        option.textContent = aName;
        selectElement.appendChild(option);
    });

    document.getElementById("overwrite-button").onclick = () => changeID(false);
    document.getElementById("add-button").onclick = () => changeID(true);
}

// Function to change the assignment ID
async function changeID(isAdd) {
    const option = selectElement.options[selectElement.selectedIndex].value;
    const assignment = assignments_info[option];
    console.log(assignment);

    if (isAdd) {
        assignment["ids"].push(assignment_id);
    } else {
        assignment["ids"] = [assignment_id];
    }

    assignment["ids"] = [...new Set(assignment["ids"])]

    try {
        await setAssignmentsInfo();
        displayMessage("Operation succeeded! Please reload the page.", true);
    } catch (error) {
        console.error("Failed to save assignment info:", error);
        displayMessage("Operation failed. Please try again.", false);
    }
}

// Function to display a short message
function displayMessage(message, success) {
    const messageDiv = document.createElement("div");
    messageDiv.textContent = message;
    messageDiv.style.position = "fixed";
    messageDiv.style.bottom = "10px";
    messageDiv.style.left = "50%";
    messageDiv.style.transform = "translateX(-50%)";
    messageDiv.style.padding = "10px";
    messageDiv.style.backgroundColor = success ? "green" : "red";
    messageDiv.style.color = "white";
    messageDiv.style.borderRadius = "5px";
    messageDiv.style.zIndex = "1000";
    document.body.appendChild(messageDiv);

    setTimeout(() => {
        messageDiv.remove(); // Remove the message after 2 seconds
    }, 2000);
}

// Main function to initialize the extension
async function main() {
    try {
        console.log("starting main");
        assignments_info = await getAssignmentsInfo();
        console.log(assignments_info);

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            currentUrl = tabs[0].url;
            console.log(currentUrl);

            const urlPattern = /^https:\/\/uiowa\.instructure\.com\/courses\/\d+\/gradebook\/speed_grader\?.*$/;
            if (!urlPattern.test(currentUrl)) {
                statusElement.textContent = "Not in Mona Lisa to Modernism ICON SpeedGrader page";
                return;
            }

            assignment_id = String(currentUrl.match(/assignment_id=(\d+)/)[1]);
            const idSpan = document.getElementById("assignment-id");
            idSpan.textContent = assignment_id;

            const nameSpan = document.getElementById("assignment-name");
            [assignment_name, assignment_entry] = findAssignmentById(assignment_id);
            if (assignment_name) {
                nameSpan.textContent = assignment_name;
                designateButton.style.display = "block";
                designateButton.onclick = () => replaceWithDesignateDiv();
                designateButton.textContent = "Redesignate current assignment"
                return;
            } else {
                statusElement.textContent = "assignment name not found";
                designateButton.style.display = "block";
                designateButton.onclick = () => replaceWithDesignateDiv();
                return;
            }
        });
    } catch (error) {
        console.error("Error in main function:", error);
        statusElement.textContent = "An error occurred. Please try again.";
    }
}

// Global variables
let assignments_info;
let currentUrl;
let assignment_id;
let assignment_name;
let assignment_entry;
const designateButton = document.getElementById("designate-button");
const designateDiv = document.getElementById("designate-div");
const selectElement = document.getElementById("select-assignment");
const statusElement = document.getElementById("status");

// Initialize the extension
main();




