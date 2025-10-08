import { useColonyStore, validateColonyData } from "../../contexts/modelStore";
import { useProfileStore } from "../../contexts/profileStore";

export const saveProfile = async (formData: {
    name: string;
    email: string;
    clan: string;
    role: string;
    picture: string;
}) => {
    console.log("Saving profile...", formData);
    const { updateProfile } = useProfileStore.getState();
    updateProfile(formData);
};

export const handleDownloadColonyData = async () => {
    const { generateColonyData } = useColonyStore.getState();
    const colonyData = generateColonyData();

    const blob = new Blob([JSON.stringify(colonyData)], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;

    a.download = "colonyData.json";
    a.click();
    URL.revokeObjectURL(url);

}

export const handleRestoreColonyData = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const { loadColonyData } = useColonyStore.getState();
    const file = event.target.files?.[0];
    if (file?.type === "application/json") {
        const text = await file.text();
        let jsonData;
        try {
            jsonData = JSON.parse(text);
        } catch (error) {
            throw new Error("Invalid JSON file");
        }

        const colonyData = validateColonyData(jsonData);
        if (!colonyData) {
            alert("Invalid colony data format.");
            return;
        } else {
            loadColonyData(colonyData);
            return;
        }

    } else {
        alert("Please upload a valid JSON file.");
        return;
    }
}