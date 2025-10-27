const KeywordType = {
    DISCUSSION: 'keywords',
    LONG: 'keywords_long',
    SHORT: 'keywords_short',
    SHORT2: 'keywords_short2'
};

class KeywordButton {
    constructor(keyword, index, type, freshlyCreated = false) {
        this.keyword = keyword;
        this.type = type;
        this.button = document.createElement("button");
        this.button.textContent = keyword;
        this.button.className = "keywords-button";
        this.button.setAttribute("data-keyword-index", index);
        this.button.style.color = highlight_colors[index];
        this.freshlyCreated = freshlyCreated;

        this.button.setAttribute("draggable", true);
        this.button.ondragstart = (event) => this.handleDragStart(event);
        this.button.ondragover = (event) => this.handleDragOver(event);
        this.button.ondrop = (event) => this.handleDrop(event);
        this.button.ondragend = () => this.handleDragEnd();

        this.button.onclick = () => this.editKeyword();
        //remove on right click
        this.button.oncontextmenu = (event) => {
            event.preventDefault();
            const column = this.button.parentElement;
            this.button.remove();
            this.updateButtonIndices(column);
        }

    }

    handleDragStart(event) {
        event.dataTransfer.setData("text/plain", this.button.getAttribute("data-keyword-index"));
        this.button.classList.add("dragging");
    }

    handleDragOver(event) {
        event.preventDefault();
        this.button.parentElement.querySelectorAll(".keywords-button").forEach(button => button.classList.remove("drag-over"));
        this.button.classList.add("drag-over");
    }

    handleDrop(event) {
        event.preventDefault();
        const draggedIndex = event.dataTransfer.getData("text/plain");
        const targetIndex = this.button.getAttribute("data-keyword-index");

        if (draggedIndex !== targetIndex) {
            this.reorderKeywords(draggedIndex, targetIndex);
        }
        this.button.classList.remove("drag-over");
    }

    handleDragEnd() {
        this.button.classList.remove("dragging");
    }

    reorderKeywords(draggedIndex, targetIndex) {
        const column = this.button.parentElement;
        const draggedButton = column.querySelector(`[data-keyword-index="${draggedIndex}"]`);
        const targetButton = column.querySelector(`[data-keyword-index="${targetIndex}"]`);

        if (draggedButton && targetButton) {
            console.log("draggedIndex: " + draggedIndex);
            console.log("targetIndex: " + targetIndex);

            if (draggedIndex < targetIndex){
                column.insertBefore(draggedButton, targetButton.nextSibling);
            } else {
                column.insertBefore(draggedButton, targetButton);
            }
            this.updateButtonIndices(column);
        }
    }

    updateButtonIndices(column) {
        const buttons = column.querySelectorAll(".keywords-button");
        buttons.forEach((button, index) => {
            console.log(button);
            console.log(index);
            button.setAttribute("data-keyword-index", index);
            button.style.color = highlight_colors[index];
        });
    }

    getIndex() {
        return this.button.getAttribute("data-keyword-index");
    }

    editKeyword() {
        const textarea = document.createElement("textarea");
        if (this.freshlyCreated) {
            textarea.value = "";
            this.freshlyCreated = false;
        } else {
            textarea.value = this.keyword;
        }
        textarea.className = "keywords-textarea";
        textarea.style.color = highlight_colors[this.getIndex()];

        textarea.addEventListener("blur", () => {
            this.updateKeyword(textarea.value);
        });

        textarea.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                this.textarea.blur();
            }
        });

        this.button.replaceWith(textarea);
        this.textarea = textarea;
        textarea.focus();
    }

    updateKeyword(newKeyword) {
        if (newKeyword.trim() !== "") {
            this.keyword = newKeyword.trim();
            this.button.textContent = this.keyword;
        }
        this.button.style.color = highlight_colors[this.getIndex()];
        this.textarea.replaceWith(this.button);
    }
}

class KeywordsPanel {
    constructor() {
        this.div = document.createElement("div");
        this.div.className = "keywords-panel";
        this.container = document.createElement("div");
        this.container.className = "keywords-container"; // Added class for container
        this.container.style.display = "none";

        this.createHeader();
        this.div.appendChild(this.container);
        this.createKeywordsContent();
        this.createFooter();
        document.body.insertAdjacentElement("beforeend", this.div);
    }

    createHeader() {
        const header = document.createElement("h1");
        header.className = "keywords-header";
        header.textContent = "KEYWORDS";

        this.arrow = document.createElement("span");
        this.arrow.textContent = "—";
        this.arrow.className = "keywords-arrow";
        this.arrow.addEventListener("click", () => { this.div.remove(); });
        header.appendChild(this.arrow);

        // Add click event to toggle container and footer visibility
        header.addEventListener("click", () => {
            if (this.container.style.display === "none") {
                this.container.style.display = "flex";
                this.footer.style.display = "block";
            } else {
                this.container.style.display = "none";
                this.footer.style.display = "none";
            }
        });

        this.div.appendChild(header);
        this.header = header;
    }

    createKeywordsContent() {
        if (assignment_type === "exam") {
            this.column1 = this.createColumn(KeywordType.SHORT, "50%", true);
            this.column2 = this.createColumn(KeywordType.LONG, "50%", true);

            if ("keywords_short2" in assignment_entry) {
                this.column1.style.width = "33%";
                this.column2.style.width = "33%"; 
                this.column3 = this.createColumn(KeywordType.SHORT2, "33%", true);
            }
        } else {
            this.column1 = this.createColumn(KeywordType.DISCUSSION, "100%", false);
        }
    }

    createColumn(type, width="100%", isExam = false) {
        const column = document.createElement("div");
        column.className = "keywords-column";
        column.style.width = width;
        column.setAttribute("data-keyword-type", type);

        const keywords = assignment_entry[type];

        if (isExam) {
            column.style.border = "1px solid #cDD8E6"; // Slight border for exam
        }

        for (let i = 0; i < keywords.length; i++) {
            const temp_button = new KeywordButton(keywords[i], i, type);
            column.insertAdjacentElement("beforeend", temp_button.button);
        }

        // Add "+ New Entry" button
        const newEntryButton = document.createElement("button");
        newEntryButton.textContent = "+";
        newEntryButton.className = "new-entry-button";
        newEntryButton.addEventListener("click", () => {
            this.addNewEntry(column, type);
        });
        column.appendChild(newEntryButton);

        this.container.appendChild(column);
        return column;
    }

    addNewEntry(column, type) {
        const index = column.children.length - 1; // Exclude the "+ New Entry" button
        const temp_button = new KeywordButton("Click here", index, type, true);
        column.insertBefore(temp_button.button, column.lastChild);
        
        assignment_entry[type].push("uninitialized");
    }

    async saveKeywords() {
        assignments_info[assignment_name] = assignment_entry;
        const columns = this.container.querySelectorAll(".keywords-column");
        columns.forEach((column) => {
            const type = column.getAttribute("data-keyword-type");
            const keywords = [];
            column.querySelectorAll(".keywords-button").forEach((button) => {
                keywords.push(button.textContent);
            });
            assignment_entry[type] = keywords;
            console.log(keywords);
        });
        // console.log("entry", assignment_entry);
        // console.log("name", assignment_name);
        // console.log("assignment", assignment_info[assignment_name])
        // console.log("assignments set" ,assignments_info);

        try {
            await setAssignmentsInfo();
            this.showFeedback("Changes saved successfully!", true);
            location.reload(true);
        } catch (error) {
            this.showFeedback("Failed to save changes: " + error.message, false);
        }
    }

    showFeedback(message, success) {
        const feedback = document.createElement("div");
        feedback.className = "feedback-message";
        feedback.textContent = message;
        feedback.style.backgroundColor = success ? "green" : "red";
        this.div.appendChild(feedback);

        setTimeout(() => {
            feedback.remove();
        }, 2000); // Remove the message after 3 seconds
    }

    createFooter() {
        this.footer = document.createElement("div");
        this.footer.className = "keywords-footer";
        this.footer.style.display = "none"; // Initially hidden

        const saveButton = document.createElement("button");
        saveButton.textContent = "Save Changes";
        saveButton.className = "save-button";
        saveButton.addEventListener("click", () => {
            this.saveKeywords();
        });

        this.footer.appendChild(saveButton);
        this.div.appendChild(this.footer);
    }
}
