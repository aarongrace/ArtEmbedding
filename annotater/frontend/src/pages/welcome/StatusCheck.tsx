import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useProfileStore } from '../../contexts/profileStore';
import { useUserStore } from '../../contexts/userStore';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { userID } = useUserStore();
  const { role } = useProfileStore();
  const navigate = useNavigate();

  useEffect(() => {
    if (!userID || role === "banned") {
      console.log("User is not logged in or is banned");
      navigate("/", { replace: true });
    }
  }, [userID, role, navigate]);

  // Render children only if the user is logged in and not banned
  if (!userID || role === "banned") {
    return null; // Prevent rendering children while redirecting
  }

  return <>{children}</>;
};

export default ProtectedRoute;
