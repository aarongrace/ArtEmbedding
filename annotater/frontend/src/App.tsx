import { useEffect, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
// import Navbar from './components/Navbar';
import { WarningBar } from './components/WarningBar';
import './globals.css';
// import Admin from './pages/admin/Admin';
// import AdminCheck from './pages/admin/AdminCheck';
import Dashboard from './pages/dashboard/Dashboard';
// import Guide from './pages/guide/guide';
// import Profile from './pages/profile/Profile';
// import StatusCheck from './pages/welcome/StatusCheck';
// import Welcome from './pages/welcome/Welcome';

function App() {

  // const [currentBg, setCurrentBg] = useState(ant_welcome);


  useEffect(() => {

  } , []);

  return (
    <div className="App">
      {/* <Navbar /> */}
      <WarningBar/>
      <main className="App-main">
        <Routes>
          <Route path="/" element={<Dashboard/>}/>
          {/* <Route path="/" element={<Welcome/>}/> */}
          {/* <Route path="/dashboard" element={<StatusCheck><Dashboard/></StatusCheck>}/> */}
          {/* <Route path="/profile" element={<StatusCheck><Profile/></StatusCheck>}/>
          <Route path="/guide" element={<StatusCheck><Guide/></StatusCheck>}/>
          <Route path="/admin" element={<AdminCheck><Admin/></AdminCheck>}/> */}
        </Routes>
      </main>
    </div>
  );
}

export default App;
