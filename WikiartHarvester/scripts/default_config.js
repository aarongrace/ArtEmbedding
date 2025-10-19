const discussion_comments = [
    {"text":'Three replies to other posts are required for each of the discussions in addition to the initial post, but you only made two.','grade': '40' },   
    {"text":'Missing two replies to other posts.','grade': '30' },   
    {"text":'You still needed to make two more replies to other posts','grade': '30' },   
    {"text":'Good work on the initial post but you still needed to do the three replies to other posts in your discussion group.','grade': '20' },   
    {"text":'Missing all the replies. Three are required each week, and they count for 10 points each.','grade': '20' },   
    {"text":'Please send me an email if you are experiencing difficulties with completing the assignments.','grade': '0' },     
    {"text":'Good work but please avoid duplicating the prompts in the initial post in the future','grade': '' }, 
    {"text":'Extension granted for late posts','grade': '' }, 
    {"text":'Great work overall but you should try to format your initial post as an essay instead of as a list.','grade': '' }, 
    {"text":'Good work on the initial post but please make sure that all of your posts meet the 175-word minimum requirement.','grade': '' },   
    {"text":'Your initial post was slightly late.','grade': '' },   
    {"text":'Late on the initial post.','grade': '' },   
    {"text":'While your posts have good information, they do not completely address all the discussion prompts. Please spend more time with them.','grade': ''},
    {"text":'Please spend more time on your responses. They need to be at least 175 words at the minimum.','grade': '' },   
]   

const default_assignments_info = {
    "Discussion 1": {
        "ids": ["2311033"],
        "type": "discussion",
        "keywords": [
            "patron, commission, Berry",
            "created, function",
            "medium, oil, manuscript",
            "naturalism",
            "symbolism, meanining, symbol, activit",
            "interesting, interested, fascinat"
        ]
    },

    "Discussion 2": {
        "ids": ["2206450"],
        "type": "discussion",
        "keywords": [
            "style, characteristics",
            "male, female, figure, nature, landscape, background, peasant, importan",
            "middleground, moment, Icarus, looking",
            "mood, time, year, day, lighting",
            "interesting, interested, fascinat, favorite"
        ]
    },

    "Discussion 3": {
        "ids": ["2206448"],
        "type": "discussion",
        "keywords": [
            "artist, movement",
            "male, female, figure, nature, landscape, background, peasant, importan",
            "middleground, moment, Icarus, looking",
            "mood, time, year, day, lighting",
            "interesting, interested, fascinat, favorite"
        ]
    },

    "Discussion 4": {
        "ids": ["1807605"],
        "type": "discussion",
        "keywords": [
            "artist, movement",
            "male, female, figure, nature, landscape, background, peasant, importan",
            "middleground, moment, Icarus, looking",
            "mood, time, year, day, lighting",
            "interesting, interested, fascinat, favorite"
        ]
    },

    "Visual Analysis Paper": {
        "ids": ["1914575"],
        "type": "essay",
        "keywords": [
            "composition",
            "subject",
            "figure, object",
            "setting, indoor, landscape",
            "room, time, day",
            "perspective, horizon, foreground, middle, background",
            "important, relate, represent",
            "colors, light, mood",
            "interesting, unusual",
            "question, artist"
        ]
    },

    "Unit Assessment 1": {
        "ids": ["2408360", "2311019"],
        "type": "exam",
        "keywords_short": [
            "medium, Ital,padua,scrovegni,arena, chapel,fresco,buon,plaster",
            "Giotto,innovation, naturalis",
            "subject",
            "donkey",
            "composition,smile,expression",
            "mood,emotion, jesus, Christ",
            "figure,disciple,follower,apostle",
            "angel",
            "foreground, landscape, setting,jerusalem,building,cityscape",
            "perspective, foreshoretening",
            "viewer, 1305,remarkable",
        ],
        "keywords_long": [
            "hours, book,january",
            "Europe, north,france,paris",
            "manuscript, illumin,vellum,ink,parchment",
            "patron,subject, duke",
            "feast, banquet, table, meal, food, drink",
            "dress,knight,movement",
            "object, armor, sword, helmet, shield",
            "foreshortening,space, perspective",
            "innovation,interesting, unusual,naturalis",
            "1412, limbourg"
        ]
    },

    "Unit Assessment 2": {
        "ids": ["2206447", "2206444"],
        "type": "exam",
        "keywords_short": [
            "genre, everyday",
            "England, North",
            "medium, oil",
            "figure, expression",
            "setting, room",
            "object, shel",
            "skull, anamorphic",
            "religio, allusion",
            "natualis",
            "humanis"
        ],
        "keywords_long": [
            "artist, Michelangelo",
            "media, medium, oil, fresco",
            "Italy",
            "narrative, Adam, Eve",
            "landscape",
            "chiaroscuro",
            "innovation",
            "interesting, unusual",
            "Leonardo, Vinci"
        ]
    },

    "Unit Assessment 3": {
        "ids": ["2206443", "2206445"],
        "type": "exam",
        "keywords_short": [
            "genre, portrait",
            "spain, South",
            "patron, philip, mariana, king, queen",
            "first, choos, chos",
            "infanta, dwar, dog",
            "studio, Rubens, mirror, alexander",
            "perspective, vanish",
            "innovat"
        ],
        "keywords_long": [
            "italy",
            "famous, woman, academy",
            "orazio, father",
            "caravaggio",
            "judith, book, assyrian",
            "gesture, expression, maid",
            "dark, setting, room, tenebrism",
            "tension, white, black, red"
        ]
    },

    "Unit Assessment 4": {
        "ids": ["2206446", "2206438"],
        "type": "exam",
        "keywords_short": [
            "genre, everyday",
            "scien",
            "figures",
            "setting",
            "light, dark, mood",
            "Enlightenment"
        ],
        "keywords_long": [
            "France",
            "Romantic",
            "history",
            "shipwreck, event",
            "figures",
            "setting",
            "sublime",
            "artist"
        ]
    },

    "Unit Assessment 5": {
        "ids": ["1807593", "1807597"],
        "type": "exam",
        "keywords_short": [
            "country, France",
            "movement, post, impressionis",
            "pointillis, divisionis",
            "figure",
            "space, perspectiv",
            "colors"
        ],
        "keywords_short2": [
            "country, spain",
            "cubis",
            "primitiv, afric, iberi, polynesi",
            "subject, nude",
            "relate, pose, gesture",
            "space",
            "characteristic, abstract"
        ],
        "keywords_long": [
            "mexico, known",
            "figure",
            "relate, expression, pose, gesture",
            "costume",
            "setting",
            "challeng, women, woman",
            "works, succeed, challeng"
        ]
    }
};