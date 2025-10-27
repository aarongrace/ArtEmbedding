import { create } from "zustand";

interface WarningStore {
    warningText: string;
    isDisplayed: boolean;
    startWarning: (text: string, time:number) => void;
}


//todo add more parameters
export const useWarningStore = create<WarningStore>((set, get)=>({
    warningText:  "",
    isDisplayed:  false,
    startWarning: (text: string, time: number=2000) =>{
        console.log("starting warning");
        if (text === get().warningText && get().isDisplayed){
            console.log("Warning already displayed");
            return;
        }
        set({warningText: text, isDisplayed: true});
        console.log(get().warningText);
        console.log(get().isDisplayed);
        setTimeout(()=>{
            set({ isDisplayed: false,
                warningText: "",
            })
        }, time)
    }
}))

export const WarningBar = ()=>{
    const { warningText, isDisplayed} = useWarningStore();

    return (
        <div id="WarningBar" style={{ 
            display: isDisplayed ? "block" : "none", 
            backgroundColor: "yellow", 
            color: "red",
            padding: "10px", 
            border: "1px solid red", 
            position: "fixed", 
            top: "30%", 
            left: "50%", 
            transform: "translate(-50%, -50%)",
            fontSize: "1.5rem",
            zIndex: 9999,
        }}>
            <h1>{warningText}</h1>
        </div>
    )
}