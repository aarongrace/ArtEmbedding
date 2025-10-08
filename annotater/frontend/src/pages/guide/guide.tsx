import { Link } from "react-router-dom";
import "./guide.css"; // Import the CSS file for styling

const Guide = () => {
  return (
    <div className="guide-container">
      <h1 className="guide-title">Game Guide</h1>
      <p className="guide-description">
        Welcome to the Clash of Colonies guide! This page will help you learn the basics of the game.
      </p>
      <ul className="guide-list">
        <li>Build and expand your colony.</li>
        <li>Gather resources like food and sand.</li>
        <li>Train ants to defend your colony.</li>
        <li>Join clans and compete on the leaderboard.</li>
        <li>Left click on food source to assign a foraging ant. Right-click it to unassign.</li>
      </ul>
      <Link to="/dashboard" className="guide-back-button">
        Back to Dashboard
      </Link>
    </div>
  );
};

export default Guide;