import axios from 'axios';
// dashboard.service.ts
// ==========================
// Placeholder service functions for dashboard actions
// ==========================

export interface PaintingData {
  id: string;
  title: string;
  artist: string;
  year: string;

  // --- Raw WikiArt metadata ---
  genre: string[];
  movement: string[];
  tags: string[];

  // --- Image & Model output ---
  imageUrl: string;      // base64-encoded full image
  vector: number[];   // embedding from model forward pass
}


/**
 * Fetches a painting (image + metadata + embedding vector) from backend.
 * Accepts an optional params dictionary for query parameters.
 */
export const fetchPainting = async (): Promise<PaintingData> => {
  try {
    const response = await axios.get("http://localhost:8000/model/painting");
    return response.data;
  } catch (error) {
    console.error("Error fetching painting:", error);
    throw error;
  }
};


/**
 * Finalizes the current annotation and fetches the next image.
 * Simulates uploading current image id and vector to backend, 
 * then retrieves the next painting.
 */
export const finalizeAndGetNext = async (
  currentData: { id: string; vector: number[] }
): Promise<PaintingData> => {
  console.log("Uploading annotation:", JSON.stringify(currentData, null, 2));

  // post the label
  const label = { id: currentData.id, vector: currentData.vector };
  await postGroundTruthLabel(label);

  // Return the next painting (placeholder)
  return await fetchPainting();
};

export const postGroundTruthLabel = async (
  labelData: { id: string; vector: number[] }
): Promise<void> => {
  try {
    const clippedVector = labelData.vector.map((val) => Math.max(0, Math.min(1, val)));
    const response = await axios.post('http://localhost:8000/model/label', { id: labelData.id, vector: clippedVector });
    console.log("Label upload response:", response.data);
  } catch (error) {
    console.error("Error uploading label:", error);
    throw error;
  }
};

export const toggleDemoMode = async ( bool: boolean): Promise<void> => {
  try {
    const response = await axios.post(`http://localhost:8000/model/set_demo_mode?value=${bool}`);
    console.log("Demo mode toggle response:", response.data);
  }
  catch (error) {
    console.error("Error toggling demo mode:", error);
    throw error;
  }
};

export const toggleLargeImagePriority = async ( bool: boolean): Promise<void> => {
  try {
    const response = await axios.post(`http://localhost:8000/model/set_fetch_large_images?value=${bool}`);
    console.log("Large image priority toggle response:", response.data);
  }
  catch (error) {
    console.error("Error toggling large image priority:", error);
    throw error;
  }
};

export const toggleFetchTestImages = async ( bool: boolean): Promise<void> => {
  try {
    const response = await axios.post(`http://localhost:8000/model/set_fetch_test_images?value=${bool}`);
    console.log("Fetch test images toggle response:", response.data);
  }
  catch (error) {
    console.error("Error toggling fetch test images:", error);
    throw error;
  }
};

export const saveModelCheckpoint = async (): Promise<void> => {
  try {
    const response = await axios.post(`http://localhost:8000/model/save_checkpoint`);
    console.log("Save model checkpoint response:", response.data);
  }
  catch (error) {
    console.error("Error saving model checkpoint:", error);
    throw error;
  } 
};