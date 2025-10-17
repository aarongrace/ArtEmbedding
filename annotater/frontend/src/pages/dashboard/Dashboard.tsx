import React, { useState } from 'react';
import './dashboard.css';

import { fetchPainting, finalizeAndGetNext, skipImage, PaintingData } from './dashboard.services'

// Slider definitions
const movements = {
  "Baroque": 0,
  "Rococo": 1,
  "Neoclassicism": 2,
  "Romanticism": 3,
  "Realism": 4,
  "Impressionism": 5,
};

const genres = {
  "Historical": 6,
  "Religious": 7,
  "Mythological": 8,
  "Everyday Life": 9,
  "Landscape": 10,
  "Portrait": 11,
};

const form = {
  "Naturalistic": 12,
  "Dynamic": 13,
  "Brushstrokes": 14,
  "Complexity": 15,
  "Balance": 16,
  "Emotionality": 17,
};

const sections = [
  { name: "Movement", sliders: movements },
  { name: "Genre", sliders: genres },
  { name: "Form", sliders: form },
];

const Dashboard: React.FC = () => {
  const [sliderValues, setSliderValues] = useState<number[]>(() => {
    const values = Array(18).fill(0);
    Object.values(form).forEach(id => (values[id] = 0.5));
    return values;
  });

  const [imageLoaded, setImageLoaded] = useState(false);
  const [currentPainting, setCurrentPainting] = useState<PaintingData>
    ({"id": "", "title": "", "artist": "", "year": "", "genre": [], "movement": [], "tags": [], "imageUrl": "", "vector": []});


  // --- Vector helpers ---
  const getSliderVector = () => sliderValues;
  const setSliderVector = (vector: number[]) => setSliderValues(vector);

  const handleLoadImage = () => {
    console.log("Loading image...");
    fetchPainting().then((data: PaintingData) => {
      console.log("Fetched painting:", data);
      // You can now store it in state if needed
      setCurrentPainting(data);
      setSliderValues(data.vector)
      setImageLoaded(true);
    });
  };

  const handleFinalize = () => {
    console.log("Finalizing current and loading next image...");
    if (!currentPainting) {
      console.warn("No current painting to finalize.");
      return;
    }
    finalizeAndGetNext({ id: currentPainting.id, vector: currentPainting.vector, })
    .then((nextPainting: PaintingData) => {
      console.log("Next painting loaded:", nextPainting);
      setCurrentPainting(nextPainting);
      setSliderValues(nextPainting.vector)
    });
  };

  const handleSkip = () => {
    console.log("Skipping current image...");
    skipImage().then((nextPainting: PaintingData) => {
      console.log("Next painting loaded:", nextPainting);
      setCurrentPainting(nextPainting);
      setSliderValues(nextPainting.vector)
    });
  };

  return (
    <div className="dashboard-container">
      {/* LEFT: Sliders */}
      <div className="sliders-bar">
        {sections.map(section => (
          <section className="section" key={section.name}>
            <div className="section-label">{section.name}</div>
            <div className="slider-column">
              {Object.entries(section.sliders).map(([label, id]) => (
                <div className="slider-wrapper" key={id}>
                  <label className="slider-label">{label}</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={sliderValues[id]}
                    className="horizontal-slider"
                    onChange={(e) => {
                      const newValues = [...sliderValues];
                      newValues[id] = parseFloat(e.target.value);
                      setSliderValues(newValues);
                    }}
                  />
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>

      {/* RIGHT: Preview + Buttons */}
      <div className="right-panel">

        <div className="image-and-metadata-row" >
          <div className="image-preview">
            {imageLoaded && currentPainting ? (
              <img
                src={`http://localhost:8000${currentPainting.imageUrl}`}
                alt={currentPainting.title}
                className="preview-image"
              />
            ) : (
              <div className="loading-placeholder" onClick={handleLoadImage}>
                <p className="click-to-load" >Click to load</p>
              </div>
            )}
          </div>
          <div className='info-container'>
            <div className="preview-info">

              <h2 className="painting-title">{currentPainting.title || "No Image Loaded"}</h2>
              <p className="painting-artist">{currentPainting.artist}</p>
              <p className="painting-year">{currentPainting.year}</p>
              <p className="painting-genres">{currentPainting.genre.join(", ")}</p>
              <p className="painting-movements">{currentPainting.movement.join(", ")}</p>
              <p className="painting-tags">{currentPainting.tags.join(", ")}</p>
            </div>
          </div>
        </div>

        <div className="button-section">
          <button onClick={() => console.log(getSliderVector())}>Get Vector</button>
          <button onClick={() => setSliderVector(Array(17).fill(0.7))}>Set All to 0.7</button>
          <button onClick={handleFinalize}>Finalize & Next Image</button>
          <button onClick={handleSkip}>Skip</button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
