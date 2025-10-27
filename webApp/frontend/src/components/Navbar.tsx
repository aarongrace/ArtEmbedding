import { CircleHelp } from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useProfileStore } from '../contexts/profileStore';
import "./navbar.css";

const Navbar = () => {
  const { role, picture } = useProfileStore();
  const location = useLocation();
  const navigate = useNavigate();

  if (location.pathname === "/") {
    return null;
  }

  const handleLogout = async () => {
    try {
      await fetch(`http://localhost:8000/profiles/logout`, {
        method: "POST",
        credentials: "include",
      });
      navigate("/");  // send back to welcome page
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  return (
    <header className="App-header">
      <img src={picture ? picture : "/ant.png"} className="App-logo" alt="logo" />
      <nav className="nav-links"> 
        <Link to="/dashboard" className="nav-link">Dashboard</Link>
        <Link to="/profile" className="nav-link">Profile</Link>
        <Link to="/ants" className="nav-link">Ants</Link>
        <Link to="/clan" className="nav-link">Clan</Link>
        <Link to="/shop" className="nav-link">Shop</Link>
        {role === "admin" && (<Link to="/admin" className="nav-link">Admin</Link>)}
        <Link to="/guide" className="nav-link">
          <CircleHelp size={20} />
        </Link>
        <button onClick={handleLogout} className="logout-btn">
          Logout
        </button>
      </nav>
    </header>
  );
};

export default Navbar;
