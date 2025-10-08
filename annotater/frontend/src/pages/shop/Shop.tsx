import { useEffect } from "react";
import { Link } from "react-router-dom";
import chitin_icon from "../../assets/imgs/chitin.png"; // Import chitin icon
import { Cosmetic, reapplyPerks, Upgrade } from "../../baseClasses/Perk";
import { useColonyStore } from "../../contexts/modelStore";
import { vars } from "../../contexts/globalVariables";
import "./shop.css"; // Import the CSS file for styling

const Shop = () => {
  // Initialize default upgrades and cosmetics
  const { perks, updateColony, chitin, putColonyInfo}  = useColonyStore();

  const upgrades: Upgrade[] = perks.filter((perk) => perk instanceof Upgrade) as Upgrade[];
  const cosmetics: Cosmetic[] = perks.filter((perk) => perk instanceof Cosmetic) as Cosmetic[];


  useEffect(() => {
    console.log("Upgrades: ", upgrades);
    console.log("Cosmetics: ", cosmetics);
  }, [perks]);



  // Handle purchase logic
  const handlePurchase = (perk: Upgrade | Cosmetic) => {
    if (perk.cost > chitin) {
      console.error("Not enough chitin to purchase this perk.");
      return;
    }
    if (perk instanceof Upgrade) {
      console.log(`Purchased upgrade: ${perk.name} for ${perk.cost} coins.`);
      perk.buy();
    } else if (perk instanceof Cosmetic) {
      console.log(`Purchased cosmetic: ${perk.name} for ${perk.cost} coins.`);
      // Add logic to deduct coins and mark the cosmetic as purchased
    }
    updateColony({
      chitin: chitin - perk.cost,
    });
    reapplyPerks(perks);
    console.log(vars.ant.workerBaseSpeed);
  };

  return (
    <div className="store-container">
      <h1 className="store-title">Game Store</h1>
      <p >
        <span className="store-description"> Exchange
          <img src={chitin_icon} alt="chitin" /> ({Math.floor(chitin)} available)  for perks!</span>
      </p>
      <p>
      </p>
      {/* Section for gameplay upgrades */}
      <div className="store-section">
        <h2 className="store-subtitle">Gameplay Upgrades</h2>
        <table className="store-table">
          <tbody>
            {upgrades.map((upgrade, index) => (
              <tr key={index} className="store-row">
                <td className="store-item-name">{upgrade.name} ({upgrade.amount}%)</td>
                <td>
                  <span className="store-item-description">{`Spend ${upgrade.cost}`} 
                    <img src={chitin_icon} alt="chitin"/></span>
                {`to increase by ${upgrade.percentagePerUpgrade}%`}</td>
                <td>
                  <button
                    className="store-buy-button"
                    onClick={() => handlePurchase(upgrade)}
                  >
                    Buy
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Section for cosmetic upgrades */}
      {cosmetics.length!== 0 && <div className="store-section">
        <h2 className="store-subtitle">Cosmetic Upgrades</h2>
        <ul className="store-list">
          {cosmetics.map((cosmetic, index) => (
            <li key={index} className="store-item">
              <span className="store-item-name">{cosmetic.getName()}</span> - {cosmetic.getCost()} Coins
              <button
                className="store-buy-button"
                onClick={() => handlePurchase(cosmetic)}
                disabled={cosmetic.isPurchased}
              >
                {cosmetic.isPurchased ? "Purchased" : "Buy"}
              </button>
            </li>
          ))}
        </ul>
      </div>}



      <Link to="/dashboard" className="store-back-button">
        Back to Dashboard
      </Link>
    </div>
  );
};

export default Shop;