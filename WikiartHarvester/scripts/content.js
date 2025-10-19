console.log("Starting ICON SpeedGrader Helper extension");

//global variables
var assignments_info;
var applicationElement;
var assignment_id;
var assignment_name;
var assignment_entry;
var assignment_type;
var process_iframe;
var keywords_panel;
var left_side_bar;

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

async function setAssignmentsInfo(){
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

// Main function to run the extension logic
async function main() {
    assignments_info = await getAssignmentsInfo();
    console.log("assignments_info retrieved from storage", assignments_info);

    if (!assignments_info) {
        console.log("assignments_info not found in storage, using default");
        assignments_info = default_assignments_info;
        setAssignmentsInfo();
    }

    let current_url = window.location.href;

    //getting the assignment id from the url
    assignment_id = String(current_url.match(/assignment_id=(\d+)/)[1]);

    [assignment_name, assignment_entry] = findAssignmentById(assignment_id);
    if (assignment_name) {
        console.log("Assignment found:", assignment_name);
    } else {
        console.log("Assignment not found");
        return false;
        // throw new Error("Assignment not found, aborting extension"); //abort
    }

    assignment_type = assignment_entry["type"];

    keywords_panel = new KeywordsPanel();
    let next_student_button = document.getElementById("next-student-button");
    process_iframe = new IFrameEditor();
    left_side_bar = new LeftBar();

    if (assignment_type == "discussion") {
        set_up_gradingbox_key_event();
    }
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

function delay(time) {
    return new Promise(resolve => setTimeout(resolve, time));
}


setTimeout(main, 5000);