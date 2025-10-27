module.exports = {

"[project]/src/app/main/page.tsx [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "default": (()=>__TURBOPACK__default__export__)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$src$2f$context$2f$playerStatsStore$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/src/context/playerStatsStore.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$zustand$2f$esm$2f$react$2f$shallow$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/zustand/esm/react/shallow.mjs [app-ssr] (ecmascript)");
'use client';
;
;
;
const Main = ()=>{
    // get current state of player stats that we care about
    const { ants, food, landLvl, maxAnts, updateStats } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$src$2f$context$2f$playerStatsStore$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"])((0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$zustand$2f$esm$2f$react$2f$shallow$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useShallow"])((state)=>({
            ants: state.ants,
            food: state.food,
            landLvl: state.landLvl,
            maxAnts: state.maxAnts,
            updateStats: state.updateStats
        }))); // need the useShallow to avoid recursive infinite loops
    const addAnt = ()=>{
        if (ants < maxAnts[landLvl] && food > 0) {
            updateStats("food", food - 1);
            updateStats("ants", ants + 1);
        }
    };
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "p-3 flex justify-center items-center flex-col",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                children: "User must be authenticated in order to be here but we haven't learned user auth yet"
            }, void 0, false, {
                fileName: "[project]/src/app/main/page.tsx",
                lineNumber: 24,
                columnNumber: 13
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("img", {
                src: "/ant.png",
                onClick: addAnt,
                className: "transition-transform duration-300 hover:scale-110 cursor-pointer"
            }, void 0, false, {
                fileName: "[project]/src/app/main/page.tsx",
                lineNumber: 25,
                columnNumber: 13
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/src/app/main/page.tsx",
        lineNumber: 23,
        columnNumber: 9
    }, this);
};
const __TURBOPACK__default__export__ = Main;
}}),
"[project]/node_modules/zustand/esm/vanilla/shallow.mjs [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "shallow": (()=>shallow)
});
const isIterable = (obj)=>Symbol.iterator in obj;
const hasIterableEntries = (value)=>// HACK: avoid checking entries type
    "entries" in value;
const compareEntries = (valueA, valueB)=>{
    const mapA = valueA instanceof Map ? valueA : new Map(valueA.entries());
    const mapB = valueB instanceof Map ? valueB : new Map(valueB.entries());
    if (mapA.size !== mapB.size) {
        return false;
    }
    for (const [key, value] of mapA){
        if (!Object.is(value, mapB.get(key))) {
            return false;
        }
    }
    return true;
};
const compareIterables = (valueA, valueB)=>{
    const iteratorA = valueA[Symbol.iterator]();
    const iteratorB = valueB[Symbol.iterator]();
    let nextA = iteratorA.next();
    let nextB = iteratorB.next();
    while(!nextA.done && !nextB.done){
        if (!Object.is(nextA.value, nextB.value)) {
            return false;
        }
        nextA = iteratorA.next();
        nextB = iteratorB.next();
    }
    return !!nextA.done && !!nextB.done;
};
function shallow(valueA, valueB) {
    if (Object.is(valueA, valueB)) {
        return true;
    }
    if (typeof valueA !== "object" || valueA === null || typeof valueB !== "object" || valueB === null) {
        return false;
    }
    if (!isIterable(valueA) || !isIterable(valueB)) {
        return compareEntries({
            entries: ()=>Object.entries(valueA)
        }, {
            entries: ()=>Object.entries(valueB)
        });
    }
    if (hasIterableEntries(valueA) && hasIterableEntries(valueB)) {
        return compareEntries(valueA, valueB);
    }
    return compareIterables(valueA, valueB);
}
;
}}),
"[project]/node_modules/zustand/esm/react/shallow.mjs [app-ssr] (ecmascript)": ((__turbopack_context__) => {
"use strict";

var { g: global, __dirname } = __turbopack_context__;
{
__turbopack_context__.s({
    "useShallow": (()=>useShallow)
});
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$zustand$2f$esm$2f$vanilla$2f$shallow$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/zustand/esm/vanilla/shallow.mjs [app-ssr] (ecmascript)");
;
;
function useShallow(selector) {
    const prev = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["default"].useRef(undefined);
    return (state)=>{
        const next = selector(state);
        return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$zustand$2f$esm$2f$vanilla$2f$shallow$2e$mjs__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["shallow"])(prev.current, next) ? prev.current : prev.current = next;
    };
}
;
}}),

};

//# sourceMappingURL=_62e4e8e4._.js.map