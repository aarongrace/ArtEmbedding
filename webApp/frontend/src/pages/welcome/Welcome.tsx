import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { startOfflineMode } from "../../contexts/modelStore";
import { useProfileStore } from "../../contexts/profileStore";
import { setUserID } from "../../contexts/userStore";
import "./welcome.css";

const Welcome = () => {
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [isLoggingIn, setIsLoggingIn] = useState(true);
  const [error, setError] = useState("");
  const { fetchProfileInfo } = useProfileStore();
  const navigate = useNavigate();

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    const { username, password } = formData;
    const res = await fetch(`http://localhost:8000/profiles/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
      credentials: "include",
    });
    const data = await res.json();
    if (res.ok) {
      const loginRes = await fetch(`http://localhost:8000/profiles/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
        credentials: "include",
      });
      if (loginRes.ok) {
        const loginData = await loginRes.json();
        setUserID(loginData.userId);
        fetchProfileInfo();
        navigate("/dashboard");
      } else {
        console.error("Error auto-logging in after signup");
      }
    } else {
      console.error("Error registering user", data);
      setError(data.detail);
      startOfflineMode();
      handlePlayAsGuest();
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    const { username, password } = formData;
    const res = await fetch(`http://localhost:8000/profiles/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
      credentials: "include",
    });
    const data = await res.json();
    if (res.ok) {
      setUserID(data.userId);
      await fetchProfileInfo();
      navigate("/dashboard");
    } else {
      console.error("Error logging in", data);
      setError(data.detail);
    }
  };

  const handlePlayAsGuest = () => {
    setUserID("guest");
    navigate("/dashboard");
  };

  return (
    <div className="welcome-container">
      <img src="/logo.png" alt="Logo" className="welcome-logo" />
      <form className="log-in-form" onSubmit={isLoggingIn ? handleLogin : handleSignUp}>
        <input
          type="text"
          placeholder="Username"
          className="log-in-input"
          onChange={(e) => setFormData({ ...formData, username: e.target.value })}
          required
        />
        <input
          type="password"
          placeholder="Password"
          className="log-in-input"
          onChange={(e) => setFormData({ ...formData, password: e.target.value })}
          required
        />
        <button type="submit" className="log-in-submit">
          {isLoggingIn ? "Login" : "Sign Up"}
        </button>
      </form>

      {isLoggingIn ? (
        <p className="disclaimer">
          Don't have an account? <span onClick={() => setIsLoggingIn(false)} className="disclaimer-link">Sign Up</span>
        </p>
      ) : (
        <p className="disclaimer">
          Already have an account? <span onClick={() => setIsLoggingIn(true)} className="disclaimer-link">Login</span>
        </p>
      )}
      <p onClick={handlePlayAsGuest} className="disclaimer">
        Or if you want to then <span className="disclaimer-link">Play as Guest</span>
      </p>

      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default Welcome;
