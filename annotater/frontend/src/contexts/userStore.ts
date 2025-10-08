import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";


type UserStore = {
  userID: string | null;
}

export const useUserStore = create<UserStore>()(persist((set) => ({
    userID: null,
}), {
    name: "user-storage", // unique name
    storage: createJSONStorage(() => sessionStorage), // Use local storage
}));


// you can import the store but it is easier just to use the functions below
export const getUserID = () => useUserStore.getState().userID;
export const setUserID = (newId : string | null) => useUserStore.setState({ userID: newId });
