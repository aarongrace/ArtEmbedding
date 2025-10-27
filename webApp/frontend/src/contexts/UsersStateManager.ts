// not in use
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import fs, { write } from "fs";
import path from "path";

function findProjectRoot(): string {
  let dir = __dirname;
  while (true){
    if (fs.existsSync(path.join(dir, '.env'))) return dir;
    const parentDir = path.dirname(dir);
    if (parentDir === dir) throw new Error("Project root with .env not found");
    dir = parentDir;
  }
}

const ROOT_DIR = findProjectRoot();
const USERS_STATE_FILE = path.join(ROOT_DIR, "metadata", "users_state.json");

function loadUsersState(): Record<string, UserState> {
  if (fs.existsSync(USERS_STATE_FILE)) {
    const data = fs.readFileSync(USERS_STATE_FILE, "utf-8");
    return JSON.parse(data);
  } else {
    console.log("Users state file not found, initializing empty state.");
    fs.mkdirSync(path.dirname(USERS_STATE_FILE), { recursive: true });
    fs.writeFileSync(USERS_STATE_FILE, JSON.stringify({}, null, 2), "utf-8");
    return {};
  }
}

function saveUsersState(users: Record<string, UserState>) {
  const dir = path.dirname(USERS_STATE_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(USERS_STATE_FILE, JSON.stringify(users, null, 2), "utf-8");
}


export type UserState = {
  seen_paintings: string[];
  labels_created: Record<string, number[]>;
};

type UsersStore = {
  currentUser: string;
  users: Record<string, UserState>;

  // Actions
  setCurrentUser: (userId: string) => void;
  ensureCurrentUser: () => void;
  addSeenPainting: (imageId: string) => void;
  addLabel: (imageId: string, labelVector: number[]) => void;
  getSeenList: () => string[];
  getLabels: () => Record<string, number[]>;
};

export const useUsersStore = create<UsersStore>()(
  persist(
    (set, get) => ({
      currentUser: "admin", // leave it as admin for now
      users: loadUsersState(),

      // Set the current user
      setCurrentUser: (userId: string) => set({ currentUser: userId }),

      // Ensure current user exists in the store
      ensureCurrentUser: () => {
        const uid = get().currentUser;
        const users = get().users;
        if (!users[uid]) {
          users[uid] = { seen_paintings: [], labels_created: {} };
          set({ users });
          saveUsersState(users);
          console.log(`Created new user state for user: ${uid}`);
        }
      },

      addSeenPainting: (imageId: string) => {
        const uid = get().currentUser;
        get().ensureCurrentUser();

        const users = get().users;
        const seen = users[uid]!.seen_paintings;
        if (!seen.includes(imageId)) {
          seen.push(imageId);
          set({ users });
          saveUsersState(users);
          console.log(`User ${uid} saw painting: ${imageId}`);
        }
      },

      addLabel: (imageId: string, labelVector: number[]) => {
        const uid = get().currentUser;
        get().ensureCurrentUser();

        const users = get().users;
        users[uid]!.labels_created[imageId] = labelVector;
        set({ users });
        saveUsersState(users);
        console.log(`User ${uid} labeled painting: ${imageId}`);
      },

      getSeenList: () => {
        const uid = get().currentUser;
        get().ensureCurrentUser();
        const users = get().users;
        return users[uid]?.seen_paintings || [];
      },

      getLabels: () => {
        const uid = get().currentUser;
        get().ensureCurrentUser();
        const users = get().users;
        return users[uid]?.labels_created || {};
      },
    }),
    {
      name: "users-storage",
      storage: createJSONStorage(() => localStorage),
    }
  )
);

// Convenience functions
export const setCurrentUser = (userId: string) => useUsersStore.getState().setCurrentUser(userId);
export const addSeenPainting = (imageId: string) => useUsersStore.getState().addSeenPainting(imageId);
export const addLabel = (imageId: string, labelVector: number[]) => useUsersStore.getState().addLabel(imageId, labelVector);
export const getSeenList = () => useUsersStore.getState().getSeenList();
export const getLabels = () => useUsersStore.getState().getLabels();
