class IFrameEditor{
    already_loading = 0
    async initializeEditor(){
        if (this.already_loading) return;
        this.already_loading = 1;
        this.retries = 9999; //how many times to check for the iframe
        await this.waitForIframe();
        console.log("iframe ready, start processing");
        assignment_type == "discussion" ? this.processDiscussion() : this.processExamEssays();
        this.already_loading = 0;
    }

    constructor(){
        //tagging the next student button so that this repeats when I move to the next student
        let next_student_button = document.getElementById("next-student-button");
        next_student_button.onclick = () => this.initializeEditor();
        //a flag to check whether the woole thing has been loaded; this is important for gamify
        this.initializeEditor();
    }

    async waitForIframe(){
        while (! this.isIframeReady()){
            console.log("waiting for iframe");
            await delay(250);
            this.retries -= 1;
            if (this.retries == 0){
                console.log("can't find iframe, no retries left");
                return;
            }
        }
        console.log("iframe ready");
        await delay(250);
    }

    isIframeReady(){
        console.log("iframe_check")
        let iframe = document.getElementById("speedgrader_iframe");
        // let iframe = document.getElementById("submission-preview-iframe");
        if (!iframe || !iframe.contentDocument){
            console.log("iframe null");
            return false;
        }

        this.iFrameDoc = iframe.contentDocument;
        let pageInitalized = this.iFrameDoc.children[0]?.children[1]?.children.length > 0;
        if (assignment_type == "exam"){
            try {
                //try to focus on the input field for the short essay
                var temp_post_fields = Array.from(this.iFrameDoc.getElementById("questions").children);
                temp_post_fields[12].children[2].getElementsByClassName("user_content quiz_response_text enhanced")[0].scrollIntoView()
            } catch{
                return false;
            }
        }
        if (pageInitalized){
            this.iframe = iframe;
        }
        return pageInitalized;
    }

    processExamEssays() {
        let icLayoutContentMain = this.iFrameDoc.getElementsByClassName("ic-Layout-contentMain")[0];
        icLayoutContentMain.style.paddingRight = "0px";
        icLayoutContentMain.style.paddingLeft = "20%";


        var postFields = Array.from(this.iFrameDoc.getElementById("questions").children);

        let shortIndex = 12;
        let short2Index = 13;
        let longIndex = "keywords_short2" in assignment_entry ? 14 : 13;
        var getPostField = (postFieldNum) => {
            return postFields[postFieldNum].children[2].getElementsByClassName("user_content quiz_response_text enhanced")[0];
        }

        this.editMessageField(getPostField(shortIndex), assignment_entry["keywords_short"]);
        this.short_input = postFields[shortIndex].getElementsByClassName("question_input")[0]
        getPostField(shortIndex).scrollIntoView();
        this.editMessageField(getPostField(longIndex), assignment_entry["keywords_long"]);
        this.long_input = postFields[longIndex].getElementsByClassName("question_input")[0]

        if ("keywords_short2" in assignment_entry) {
            this.editMessageField(getPostField(short2Index), assignment_entry["keywords_short2"]);
            this.short2_input = postFields[short2Index].getElementsByClassName("question_input")[0];
        }

    }

    processDiscussion() {
        let submissionDescription = this.iFrameDoc.getElementsByClassName("submission_description")[0];
        submissionDescription.style.paddingLeft = "10%";
        this.notMeetingWordCountBy = 0;
        console.log("fasdfdsfdsfdsfsdchildren[0]children[0]")
        console.log("fasdfdsfdsfdsfsdchildren[0]children[0]")
        console.log("fasdfdsfdsfdsfsdchildren[0]children[0]")
        console.log(this.iFrameDoc)
        let content = this.iFrameDoc.getElementById("content")
        console.log(content)
        console.log(content.children)
        console.log(submissionDescription.children)
        console.log("fasdfdsfdsfdsfsdchildren[0]children[0]")
        const all_posts_including_replies = submissionDescription.children; // or whatever parent element
        this.postFields = Array.from(all_posts_including_replies).filter(el =>
            el.id.startsWith("entry_")
        );
        console.log(this.postFields)

        // this.post_fields = Array.from(this.iFrameDoc.getElementById("content").children[0].children)

        // this.post_fields = this.post_fields.filter(child => {
        //     return child.tagName == "DIV" && child.className != "" && !child.className.includes("subtopic");
        // });

        this.postFields.forEach( postField => {
            let messageFields = postField.getElementsByClassName("message user_content enhanced")
            let messageField = messageFields[0]
            this.editMessageField(messageField, assignment_entry["keywords"]);
        })
        if (this.postFields.length < 4 || this.notMeetingWordCountBy){
            this.addDiscusionWarning();
        }
    }

    addDiscusionWarning(){
        let num_of_posts = this.postFields.length;
        const infoDiv = document.createElement("div");
        infoDiv.style.textAlign = "center";
        infoDiv.style.fontSize = "20px";
        infoDiv.style.fontWeight = "bold";
        infoDiv.style.margin = "10px 0";
        infoDiv.style.padding = "10px";
        infoDiv.style.backgroundColor = "#f0f8ff";
        infoDiv.style.border = "2px solid #add8e6";
        infoDiv.style.borderRadius = "5px";
        infoDiv.style.color = "red";
        if (this.notMeetingWordCountBy) {
            const wordCountInfo = document.createElement("p");
            wordCountInfo.textContent = "Not meeting word count requirement by " + this.notMeetingWordCountBy + " words.";
            infoDiv.appendChild(wordCountInfo);
        }
        if (num_of_posts < 4) {
            const postsInfo = document.createElement("p");
            postsInfo.textContent = `Number of posts: ${num_of_posts}`;
            infoDiv.appendChild(postsInfo);
        }
        this.iFrameDoc.getElementById("content").children[0].after(infoDiv);
    }

    //adding color and word count and potentially warnings for discussions
    editMessageField(message_field, keywords){
        //Every big container has two "message user_content enhanced fields". The second one is always null
        if (message_field === undefined){
            console.log("no message field");
            return null;
        }

        var alltext = ""
        var tempchildren = message_field.children

        //there is one empty messagebox for every useful one  
        if (tempchildren === undefined || tempchildren.length == 0 ){
            console.log("nothing to see here")
            return null;
        }

        //each of the tempchildren is a <p> element in the discussion field
        //we are adding all the words and marking the ones to be marked
        for (const k in tempchildren){
            alltext += tempchildren[k].textContent;

            let temptext = tempchildren[k].textContent;
            if (temptext){
                for (let i = 0; i <keywords.length; i++){
                    let keywords_list = keywords[i].split(",").map( word => word.trim());

                    //get capitalized version of each word
                    keywords_list.forEach( keyword =>{
                        let newText = "";
                        let isInsideTag = false;

                        for (let k =0; k<temptext.length; k++){
                            if (temptext[k] == "<") isInsideTag = true;
                            if (temptext[k] == ">") isInsideTag = false;

                            if (!isInsideTag && temptext.substring(k, k + keyword.length).toLowerCase() == keyword.toLowerCase()){
                                let original_word = temptext.substring(k, k + keyword.length);
                                newText += `<span style="color:${highlight_colors[ i < 20 ? i : i % 20 ]}">${original_word}</span>`;
                                k += keyword.length - 1;
                            } else {
                                newText += temptext[k];
                            }
                        }
                        temptext = newText;
                    });
                }
                tempchildren[k].innerHTML = temptext;
            }
        }

        // the companion text ////////////////////////////////////////////////////////
        if (assignment_type == 'discussion'){
            //calculating the word count
            let wordMatchRegExp = /[^\s]+/g;
            let words = alltext.matchAll(wordMatchRegExp);
            let word_count = [...words].length;
            let word_requirement = 175; 

            var companion_string = document.createElement("p");
            companion_string.style.backgroundColor = "#FFEEFF";
            companion_string.style.color = "#7B68EE";
            companion_string.style.fontFamily = "Trebuchet MS, Helvetica, sans-serif";
            companion_string.textContent = "(SpeedGrader Helper) Word count: " + word_count + ".";
            if (word_count < word_requirement){
                this.notMeetingWordCountBy += word_requirement - word_count;
                companion_string.style.color = "red"
            }
            tempchildren[0].before(companion_string)
        }

    }

    async change_essay_grade(grade, is_long, second_short_essay = false){
        let curr_input;
        if (is_long) {
            curr_input = this.long_input
        } else if (second_short_essay) {
            curr_input = this.short2_input;
        } else { 
            curr_input = this.short_input; 
        }

        curr_input.value = grade
        let inputEvent = new Event('input', {bubbles: true});
        curr_input.dispatchEvent(inputEvent);
        let changeEvent = new Event('change', {bubbles: true});
        curr_input.dispatchEvent(changeEvent );
       
        if (!is_long) {
            if ("keywords_short2" in assignment_entry) {
                if (second_short_essay) {
                    this.long_input.scrollIntoView();
                } else {
                    this.short2_input.scrollIntoView();
                }
            } else {
                this.long_input.scrollIntoView();
            }
        } else {
            this.iframe.contentDocument.getElementsByClassName("btn btn-primary update-scores")[0].click();

            await delay(3000); // wait for the score to be updated
            // potentially add a check to see if the score has been updated instead of using a set delay
            // we can just check if the score has been updated by checking the score number
            // which is to say save the score before and after the click and compare them
            document.getElementById("next-student-button").click();
        }
    }
}

const highlight_colors = [
    "#228B22",  // Beautiful Tree Green
    "#FFD700",  // Gold Yellow
    "#FF6347",  // Tomato (Pleasant Red)
    "#008B8B",  // Dark Cyan
    "#0000FF",  // Pure Blue
    "#8A2BE2",  // Blue-Violet
    "#FF1493",  // Deep Pink
    "#A52A2A",  // Brown
    "#FF8C00",  // Dark Orange
    "#20B2AA",  // Light Sea Green
    "#1E6F1E",  // Deeper Tree Green
    "#FFC000",  // Warmer Gold Yellow
    "#2EAD2E",  // Deeper Lime Green
    "#007777",  // Darker Cyan
    "#0000E0",  // Slightly Darker Blue
    "#7A1FD1",  // More Muted Blue-Violet
    "#E01185",  // Richer Deep Pink
    "#8B1E1E",  // Darker Brown
    "#E67800",  // Slightly Darker Orange
    "#1E978F"   // Darker Light Sea Green
];
