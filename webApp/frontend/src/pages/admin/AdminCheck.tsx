import React from 'react';
import { Navigate } from 'react-router-dom';
import { useProfileStore } from '../../contexts/profileStore';

interface AdminRouteProps {
  children: React.ReactNode;
}

const AdminRoute: React.FC<AdminRouteProps> = ({children}) => {
  const {role} = useProfileStore();

  if (role != "admin" ) {
    return <Navigate to="/dashboard" replace />;
  }
  return <>{children}</>;
};

export default AdminRoute;
