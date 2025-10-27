import { create } from "zustand";
import { vars } from "./globalVariables";
import { useUserStore } from "./userStore";

type ProfileData = {
  id: string;
  name: string;
  email: string;
  clan: string;
  role: string;
  picture: string;
}

type ProfileStore = {
  id: string;
  name: string;
  email: string;
  clan: string;
  role: string;
  picture: string;
  fetchProfileInfo: () => Promise<void>;
  updateProfile: (profile: Partial<ProfileData>)=> Promise<void>;
};

export const useProfileStore = create<ProfileStore>((set) => ({
  id: "",
  name: "",
  email: "",
  clan: "",
  role: "user",
  picture: "",

  // Fetch a profile from the backend
  fetchProfileInfo: async () => {
    if (vars.offline_mode){
      console.warn("In offline mode, and will not fetch profile info");
      return;
    }

    const userID = useUserStore.getState().userID; // Access userID dynamically
    if (!userID) {
      throw new Error("User ID is not set");
    }

    var response;
    try {
      response = await fetch(`http://localhost:8000/profiles/${userID}`);
    } catch (e) {
      console.warn("Failed to fetch profile:", e);
      vars.offline_mode = true;
    }

    if (!response || !response.ok) {
      throw new Error("Failed to fetch profile information");
    }

    const data = await response.json();
    console.log("Profile data:", data);
    set({
      id: data['_id'],
      name: data.name,
      email: data.email,
      clan: data.clan,
      role: data.role,
      picture: data.picture,
    });
  },

  updateProfile: async (profileData: Partial<ProfileData>) => {
    console.log("Updating profile with data:", profileData);
    if (vars.offline_mode){
      console.warn("In offline mode and cannot update profile")
      return;
    }

    const userID = useUserStore.getState().userID;
    try {
      const response = await fetch(`http://localhost:8000/profiles/${userID}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(profileData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Network response was not ok");
      }

      const data = await response.json();
      console.log("Profile updated successfully:", data);

    } catch (error) {
      if (error instanceof Error) {
        console.error("Error saving profile:", error.message);
      } else {
        console.error("Error saving profile:", error);
      }
    }
  },

}));