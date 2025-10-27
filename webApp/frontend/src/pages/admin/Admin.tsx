import React, { useState } from "react";
import { Link } from "react-router-dom";
import { updateUserRole, deleteUserProfile, getData, getColony, getTrades } from "./admin.services";
import "./admin.css";

const Admin: React.FC = () => {
  const [username, setUsername] = useState("");

  const handleRoleChange = (newRole: string) => async (
    e: React.MouseEvent<HTMLButtonElement>
  ) => {
    e.preventDefault();
    if (!username.trim()) {
      alert("Enter a username");
      return;
    }
    try {
      const updatedUser = await updateUserRole(username, newRole);
      alert(`User "${updatedUser.name}" updated to role "${updatedUser.role}"`);
    } catch (error: any) {
      console.error("Error updating role:", error);
      if (error.message.includes("Profile not found")) {
        alert("User with the entered name was not found.");
      } else {
        alert("Error updating role: " + error.message);
      }
    }
  };

  const handleDelete = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    if (!username.trim()) {
      alert("Enter a username");
      return;
    }
    if (
      !window.confirm(
        "Are you sure you want to delete this profile? This action cannot be undone."
      )
    ) {
      return;
    }
    try {
      const result = await deleteUserProfile(username);
      alert(result.message);
    } catch (error: any) {
      console.error("Error deleting profile:", error);
      if (error.message.includes("Profile not found")) {
        alert("User with the entered name was not found.");
      } else {
        alert("Error deleting profile: " + error.message);
      }
    }
  };

  const handleGetData = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    if (!username.trim()) {
      alert("Please enter a username");
      return;
    }
    try {
      const Info = await getData(username);
      alert(JSON.stringify(Info, null, 2));
    } catch (error: any) {
      console.error("Error displaying user info:", error);
    }
  };

  const handleGetColony = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    if (!username.trim()) {
      alert("Please enter a username");
      return;
    }
    try {
      const Info = await getColony(username);
      alert(JSON.stringify(Info, null, 2));
    } catch (error: any) {
      console.error("Error displaying user info:", error);
    }
  };

  const handleGetTrades = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    if (!username.trim()) {
      alert("Please enter a username");
      return;
    }
    try {
      const Info = await getTrades(username);
      alert(Info);
    } catch (error: any) {
      console.error("Error displaying pending trades:", error);
    }
  };

  const roleButtons = [
    { label: "Promote", role: "admin" },
    { label: "Set as User", role: "user" },
    { label: "Ban", role: "banned" },
  ];

  return (
    <div className="admin-container">
      <h1 className="admin-title">Admin Panel</h1>
      <p className="admin-description">
        Manage players with one convenient interface.
      </p>
      <form className="admin-form">
        <input
          type="text"
          placeholder="Player Username"
          className="admin-input"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <div className="admin-buttons">
          {roleButtons.map(({ label, role }) => (
            <button
              key={`${role}-${label}`}
              onClick={handleRoleChange(role)}
              className="admin-submit"
              type="button"
            >
              {label}
            </button>
          ))}
          <button onClick={handleDelete} className="admin-submit" type="button">
            Delete User
          </button>
          <div>
            <button onClick={handleGetData} className="admin-submit" type="button">
              Profile Data
            </button>
            <button onClick={handleGetColony} className="admin-submit" type="button">
              Colony Data
            </button>
            <button onClick={handleGetTrades} className="admin-submit" type="button">
              Pending trades
            </button>
          </div>
        </div>
      </form>
      <Link to="/dashboard" className="admin-back-button">
        Back to Dashboard
      </Link>
    </div>
  );
};


export default Admin;
