class DiscussionButton {
    constructor(grade) {
        this.button = document.createElement("button");
        this.button.textContent = grade;
        this.button.className = "discussion-button";
        this.button.addEventListener("click", (event) => { this.changegrade(grade) })
    }

    changegrade(grade) {
        let gradingbox = document.getElementById("grading-box-extended")
        gradingbox.select()
        gradingbox.value = String(grade)

        let inputEvent = new Event('input', {bubbles: true});
        gradingbox.dispatchEvent(inputEvent);
        let changeEvent = new Event('change', {bubbles: true});
        gradingbox.dispatchEvent(changeEvent );

        document.getElementById("next-student-button").click();
    }
}

class EssayButton {
    constructor(grade, is_long, second_short_essay = 0) {
        let has_second_short_essay = "keywords_short2" in assignment_entry;
        let max_grade = is_long ? 46 : (has_second_short_essay ? 15 : 30);
        this.button = document.createElement("button")
        this.button.textContent = grade + " (" + (parseFloat(grade) * 100 / max_grade).toFixed(1).toString() + ")"
        this.button.className = "essay-button";
        this.button.addEventListener("click", (event) => { 
            process_iframe.change_essay_grade(grade, is_long, second_short_essay) 
        })
    }
}

class LeftBar {
    constructor() {
        this.bar = document.createElement("div")
        this.bar.className = "left-bar";
        document.body.insertAdjacentElement("beforeend", this.bar)

        var styleElement = document.createElement("style");
        styleElement.appendChild(document.createTextNode("div ::-webkit-scrollbar {-webkit-appearance: none;width: 0px;}div ::-webkit-scrollbar-thumb {border-radius: 0px;}"));
        this.bar.appendChild(styleElement);

        if (assignment_type == "exam") {
           this.bar.style.width = "19%";
        } else {
            this.bar.style.width = "19%";
        }

        if (assignment_type == "discussion") {
            this.createDiscussionButtons();
        } else {
            this.createEssayButtons(false);
            if ("keywords_short2" in assignment_entry) {
                this.createEssayButtons(false, true);
            }
            this.createEssayButtons(true);
        }
    }

    createDiscussionButtons() {
        this.list_of_comments = discussion_comments
        this.header = document.createElement("d")
        this.header.textContent = "Grades"
        this.header.className = "left-bar-header";
        this.bar.appendChild(this.header)

        this.buttons_div = document.createElement("div")
        this.bar.appendChild(this.buttons_div)
        let min_grade = 10;
        let max_grade = 50;
        let grades_num = [...Array(max_grade * 2 - min_grade * 2 + 1).keys()].map(i => i * 0.5 + min_grade);
        //console.log(grades_num);
        grades_num = grades_num.concat([...Array(min_grade / 2 - 1).keys()].map(i => i + min_grade / 2 + 1));
        grades_num.sort((a, b) => b - a)
        grades_num[grades_num.length-1] = 0;
        for (let num of grades_num) {
            let tempbutton = new DiscussionButton(num)
            this.buttons_div.appendChild(tempbutton.button)
        }
    }

    createEssayButtons(is_long, second_short_essay = 0) {
        let header = document.createElement("p");
        let header_text = is_long ? "long" : "short";
        let has_second_short_essay = "keywords_short2" in assignment_entry;
        header.textContent = header_text.charAt(0).toUpperCase() + header_text.slice(1)

        if (second_short_essay == 1) {
            header.textContent += "2"
        }

        header.className = "left-bar-section-header";
        this.bar.appendChild(header)

        let min_grade = is_long ? 36 : (has_second_short_essay ? 12 : 24);
        let max_grade = is_long ? 46 : (has_second_short_essay ? 15 : 30);
        console.log("min_grade", min_grade, "max_grade", max_grade)


        let grades_num = [...Array(max_grade * 2 - min_grade * 2 + 1).keys()].map(i => i * 0.5 + min_grade);
        grades_num = grades_num.concat([...Array(min_grade / 2 - 1).keys()].map(i => i + min_grade / 2 + 1));
        grades_num.sort((a, b) => b - a);

        for (let k in grades_num) {
            let tempbutton = new EssayButton(grades_num[k], is_long, second_short_essay = second_short_essay)
            this.bar.appendChild(tempbutton.button)
        }
    }
}

function set_up_gradingbox_key_event() {

    var gradingbox = document.getElementById("grading-box-extended")

    console.log(gradingbox)
    gradingbox.addEventListener('keydown', function (event) {
        const key = event.key; // "a", "1", "Shift", etc.
        if (key == " ") {

            delay(2000).then(() => {

                gradingbox.click()
                document.getElementById("comment_submit_button").click()

                let next_student_button = document.getElementById("next-student-button")
                next_student_button.click()
            })
        }
    });
}