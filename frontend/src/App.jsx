import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import { ToastContainer } from "react-toastify";
import { Toaster } from "react-hot-toast";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public */}
        <Route path="/" element={<LandingPage />} />
      </Routes>
      <ToastContainer/>
      <Toaster
  position="top-center"
  reverseOrder={false}
/>
    </BrowserRouter>
  );
}