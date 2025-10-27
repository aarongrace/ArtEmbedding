import { useProfileStore } from "../../contexts/profileStore";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  clan: string;
  role: string;
  picture: string;
}

export const saveProfile = async (profileData: UserProfile): Promise<UserProfile> => {
  console.log("Saving profile...", profileData);
  const { updateProfile } = useProfileStore.getState();
  updateProfile(profileData);
  return profileData;
};

export const updateUserRole = async (
  username: string,
  newRole: string
): Promise<UserProfile> => {
  const response = await fetch(`${API_BASE_URL}/profiles/update-role/${username}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ role: newRole }),
  });

  if (!response.ok) {
    let errorMsg = "Error updating role";
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail || errorMsg;
    } catch (err) {
      console.error("Failed to parse error response", err);
    }
    throw new Error(errorMsg);
  }

  const updatedProfile: UserProfile = await response.json();

  return updatedProfile;
};

export const deleteUserProfile = async (
  username: string
): Promise<{ message: string }> => {
  const response = await fetch(`${API_BASE_URL}/profiles/delete/${username}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    let errorMsg = "Error deleting profile";
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail || errorMsg;
    } catch (err) {
      console.error("Failed to parse error response", err);
    }
    throw new Error(errorMsg);
  }

  const result = await response.json();
  return result;
};


export const getData = async (
  username: string
): Promise<{ message: string }> => {
  const response = await fetch(`${API_BASE_URL}/profiles/get-data/${username}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    let errorMsg = "Error getting profile";
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail || errorMsg;
    } catch (err) {
      console.error("Failed to parse error response", err);
    }
    throw new Error(errorMsg);
  }

  const result = await response.json();
  return result;
};


export const getColony = async (username: string): Promise<any> => {

  const idResponse = await fetch(`${API_BASE_URL}/profiles/get-id/${username}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  const idData = await idResponse.json();
  const userId = idData.id;

  const colonyResponse = await fetch(`${API_BASE_URL}/colonies/${userId}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  
  if (!colonyResponse.ok) {
    let errorMsg = "Error getting colony";
    try {
      const errorData = await colonyResponse.json();
      errorMsg = errorData.detail || errorMsg;
    } catch (err) {
      console.error("Failed to parse error response", err);
    }
    throw new Error(errorMsg);
  }
  
  const result = await colonyResponse.json();
  return result;
};

export const getTrades = async (username: string): Promise<any> => {

  const idResponse = await fetch(`${API_BASE_URL}/profiles/get-id/${username}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  const idData = await idResponse.json();
  const userId = idData.id;

  const colonyResponse = await fetch(`${API_BASE_URL}/trades/sentPending/${userId}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  
  if (!colonyResponse.ok) {
    let errorMsg = "Error getting trades";
    try {
      const errorData = await colonyResponse.json();
      errorMsg = errorData.detail || errorMsg;
    } catch (err) {
      console.error("Failed to parse error response", err);
    }
    throw new Error(errorMsg);
  }
  
  const result = await colonyResponse.json();

  const colonyResponse2 = await fetch(`${API_BASE_URL}/trades/pending/${userId}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  
  if (!colonyResponse2.ok) {
    let errorMsg = "Error getting trades";
    try {
      const errorData = await colonyResponse2.json();
      errorMsg = errorData.detail || errorMsg;
    } catch (err) {
      console.error("Failed to parse error response", err);
    }
    throw new Error(errorMsg);
  }
  
  const result2 = await colonyResponse2.json();

  return "to: " + JSON.stringify(result, null, 2) + " from: " + JSON.stringify(result2, null, 2);
};
