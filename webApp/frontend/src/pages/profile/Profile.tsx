import React, { useEffect, useState } from 'react';
import { useProfileStore } from '../../contexts/profileStore';
import './Profile.css';
import { handleDownloadColonyData, handleRestoreColonyData, saveProfile } from './profile.services';

const Profile: React.FC = () => {
  const { id, name, email, clan, role, picture, fetchProfileInfo } = useProfileStore();
  const [clanName, setClanName] = useState<string>('');
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    clan: '',
    role: '',
    picture: '',
    password: '', // Add password to formData
  });

  const backupInputJSONRef = React.useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchProfileInfo().then(async () => {
      setFormData({
        name: name,
        email: email,
        clan: clan,
        role: role,
        picture: picture,
        password: '', // Initialize password
      });
  
      if (clan) {
        try {
          const res = await fetch(`http://localhost:8000/clan/${clan}`, {
            credentials: "include",
          });
          const data = await res.json();
          setClanName(data.name);
        } catch (error) {
          console.error("Error fetching clan name:", error);
        }
      } else {
        setClanName(""); 
      }
    });
  }, [fetchProfileInfo, name, email, clan, role, picture]);

  const handleSave = async () => {
    console.log('Saving profile data:', formData);
    await saveProfile(formData);
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setFormData({
          ...formData,
          picture: reader.result as string, // base64 string
        });
      };
      reader.readAsDataURL(file); // Converts to base64
    }
  };

  return (
    <div className="profile-container">
      <h1 className="profile-header">{name} Profile</h1>
      <form className="profile-form">
        <div className="form-group">
          <label htmlFor="name">Name:</label>
          <input
            type="text"
            id="name"
            name="name"
            className="profile-input"
            value={formData.name}
            onChange={(e) => setFormData({...formData, name: e.target.value})}
          />
        </div>
        <div className="form-group">
          <label htmlFor="email">Email:</label>
          <input
            type="email"
            id="email"
            name="email"
            className="profile-input"
            value={formData.email}
            onChange={(e) => setFormData({...formData, email: e.target.value})}
            placeholder="user@email.com"
          />
        </div>

        <div className="form-group">
          <label htmlFor="clan">Clan:</label>
          <input
            type="text"
            id="clan"
            name="clan"
            className="profile-input"
            value={clan ? clanName : "No Clan"}
            disabled // cant edit clan either must go to /clan to handle that stuff
          />
        </div>
        <div className="form-group">
          <label htmlFor="role">Role:</label>
          <input
            type="text"
            id="role"
            name="role"
            className="profile-input"
            value={formData.role}
            disabled // Role is typically not editable
          />
        </div>
        <div className="form-group">
        <label htmlFor="picture">Profile Picture:</label>
          {picture && (
            <img src={formData.picture} alt="Profile" className="profile-picture" style={{width: '100px', height: '100px'}}/>
          )}
          <input
            type="file"
            id="picture"
            name="picture"
            className="profile-input"
            accept="image/*"
            onChange={handleImageChange}
            style={{ color: 'black' }}
          />
        </div>
        <div className="form-group">
          <label htmlFor="password">Password:</label>
          <input
            type="password"
            id="password"
            name="password"
            className="profile-input"
            value={formData.password}
            onChange={(e) => setFormData({...formData, password: e.target.value})}
            placeholder="Enter your password"
          />
        </div>
        <button type="button" className="profile-button" onClick={handleSave}>
          Save
        </button>
      </form>

      {/* New Buttons for Download and Restore Colony Data */}
      <div className="profile-actions">
        <button
          type="button"
          className="profile-action-button"
          onClick={handleDownloadColonyData}
        >
          Download Colony Data
        </button>
        <button
          type="button"
          className="profile-action-button"
          onClick={() => {backupInputJSONRef.current?.click();}}
        >
          Restore Colony Data
        </button>
        <input type="file" ref={backupInputJSONRef} style = {{ display: 'none' }} onChange={handleRestoreColonyData} />
      </div>
    </div>
  );
};

export default Profile;