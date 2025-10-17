import { get } from 'lodash';
import sampleImage from '../../assets/imgs/013106_edwin-henry-landseer—flood-in-the-highlands.jpg' 
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
 * To be implemented with actual FastAPI endpoint.
 */
export const fetchPainting = async (): Promise<PaintingData> => {
  try {
    const response = await axios.get('http://localhost:8000/model/painting');
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
  console.log("Service: finalizeAndGetNext() called.");
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
    const response = await axios.post('http://localhost:8000/model/label', labelData);
    console.log("Label upload response:", response.data);
  } catch (error) {
    console.error("Error uploading label:", error);
    throw error;
  }
};

/**
 * Skips the current image and moves to the next.
 * Currently identical to finalizeAndGetNext placeholder.
 */
export const skipImage = async (): Promise<PaintingData> => {
  console.log("Service: skipImage() called — skipping current image.");
  await new Promise((resolve) => setTimeout(resolve, 300)); // simulate latency
  return await fetchPainting();
};
