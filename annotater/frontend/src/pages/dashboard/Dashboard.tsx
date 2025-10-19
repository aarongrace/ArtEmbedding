import React, { useState } from 'react';
import './dashboard.css';

import { fetchPainting, finalizeAndGetNext, PaintingData, saveModelCheckpoint, toggleDemoMode, toggleLargeImagePriority as toggleFetchLargeImages, toggleFetchTestImages } from './dashboard.services'

// Slider definitions
const movements = {
  "Baroque": 0,
  "Rococo": 1,
  "Neoclassicism": 2,
  "Romanticism": 3,
  "Realism": 4,
  "Impressionism": 5,
};

// used to clip years for movements; broader to allow some tolerance
const movementsTimeRange = {
  "Baroque": [1580, 1750],
  "Rococo": [1710, 1820],
  "Neoclassicism": [1750, 1860],
  "Romanticism": [1800, 1860],
  "Realism": [1830, 1900],
  "Impressionism": [1850, 1920],
}

const genres = {
  "Historical": 6,
  "Religious": 7,
  "Mythological": 8,
  "Everyday Life": 9,
  "Landscape": 10,
  "Portrait": 11,
};

const form = {
  "Balance": 12,
  "Complexity": 13,
  "Emotionality": 14,
  "Dynamic": 15,
  "Naturalistic": 16,
  "Brushstrokes": 17,
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
    ({id: "", title: "", artist: "", year: "", genre: [], movement: [], tags: [], imageUrl: "", vector: []});
  
  const [demoMode, setDemoMode] = useState(false);
  const [fetchTestImages, setFetchTestImages] = useState(false);

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
    if (currentPainting.id === "") {
      console.warn("No current painting to finalize.");
      handleLoadImage();
      return;
    }
    finalizeAndGetNext({ id: currentPainting.id, vector: sliderValues })
    .then((nextPainting: PaintingData) => {
      console.log("Next painting loaded:", nextPainting);
      setCurrentPainting(nextPainting);
      setSliderValues(nextPainting.vector)
    });
  };

  const handleDemoToggle = (value : boolean) => {
    if (fetchTestImages === true && value === false) {
      alert("Demo mode must be enabled when fetching test images, otherwise we would contaminate the test set.");
      setDemoMode(true);
      return;
    }
    console.log("Toggling demo mode to:", value);
    setDemoMode(value);
    toggleDemoMode(value);
  }

  const handleFetchTestImagesToggle = (value: boolean) => {
    console.log("Toggling fetch test images to:", value);
    toggleFetchTestImages(value);
    setFetchTestImages(value);
    if (value === true && demoMode === false) {
      setDemoMode(true);
      toggleDemoMode(true);
      console.log("Enabling demo mode as well since test images are being fetched.");
    }
  };

  const handleClipSmallValues = () =>{
    console.log("Clipping small slider values to zero.");
    let currSliderValues = [...sliderValues]
    for (let i = 0; i < Object.entries(movements).length + Object.entries(genres).length; i++) {
      if (currSliderValues[i]! < 0.2) {
        currSliderValues[i] = 0;
      }
    }
    setSliderValues(currSliderValues);
  }

  const handleClipMovement = () =>{
    let yearString = currentPainting.year;
    if (yearString.includes("-") && yearString.length > 4) {
      yearString = yearString.split("-")[0]!;
    }
    
    const currYear = parseInt(yearString, 10);
    if (isNaN(currYear)) {
      console.warn("Current painting year is not a valid number:", currentPainting.year);
      return;
    }
    console.log("Clipping movement sliders based on year:", currYear);
    let currSliderValues = [...sliderValues]
    for (const [movementKey, index] of Object.entries(movements)) {
      const movement = movementKey as keyof typeof movementsTimeRange;
      const [startYear, endYear] = movementsTimeRange[movement];
      if (currYear < startYear! || currYear > endYear!) {
        currSliderValues[index] = 0;
      }
    }
    setSliderValues(currSliderValues);

  };

  const handleSkip = () => {
    console.log("Skipping current image...");
    handleLoadImage();
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
              <p className="painting-artist">Artist: <strong>{currentPainting.artist}</strong></p>
              <p className="painting-year">Year: <strong>{currentPainting.year}</strong></p>
              <p className="painting-genres">Genres: <strong>{currentPainting.genre.map(genre => genre.charAt(0).toUpperCase() + genre.slice(1)).join(", ")}</strong></p>
              <p className="painting-movements">Styles: <strong>{currentPainting.movement.map(style => style.charAt(0).toUpperCase() + style.slice(1)).join(", ")}</strong></p>
              {/* <p className="painting-tags">Tags: <strong>{currentPainting.tags.join(", ")}</strong></p> */}
            </div>
          </div>
        </div>

        <div className="button-section">
          <button onClick={handleFinalize}> Submit & Next</button>
          <button onClick={handleSkip}>Skip</button>
          <button onClick={saveModelCheckpoint}>Save Model</button>
          <button onClick={handleClipMovement}>Clip Movement Vals</button>
          <button onClick={handleClipSmallValues}>Clip Small Values</button>
        </div>
        <div className="toggles-section">
          <label className ="toggle-label">
            <input type="checkbox" defaultChecked={true} onChange={(e) => toggleFetchLargeImages(e.target.checked)} />
            Get Large Images
          </label>
          <label className ="toggle-label">
            <input type="checkbox" checked={demoMode} onChange={(e) => handleDemoToggle(e.target.checked)} />
            Demo Mode (not saving)
          </label>
          <label className ="toggle-label">
            <input type="checkbox" checked={fetchTestImages} onChange={(e) => handleFetchTestImagesToggle(e.target.checked)} />
            Get Test Images
          </label>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
