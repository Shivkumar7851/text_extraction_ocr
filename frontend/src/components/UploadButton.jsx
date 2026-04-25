import { useRef } from "react";

export default function UploadButton() {
  const fileInputRef = useRef();

  const handleClick = () => {
    fileInputRef.current.click(); // open file dialog
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    console.log(file); // you can send this to backend
  };

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Hidden Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Styled Button */}
      <button
        onClick={handleClick}
        className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg shadow"
      >
        Upload File
      </button>
    </div>
  );
}