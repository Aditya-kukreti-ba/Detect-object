import React, { useState } from 'react';
import { Upload, Camera, Loader2, AlertCircle, CheckCircle } from 'lucide-react';

export default function YOLOSObjectDetector() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [backendStatus, setBackendStatus] = useState('unknown');

  // Check if backend is running
  const checkBackend = async () => {
    try {
      const response = await fetch('http://localhost:5000/health');
      if (response.ok) {
        setBackendStatus('online');
      } else {
        setBackendStatus('offline');
      }
    } catch (err) {
      setBackendStatus('offline');
    }
  };

  React.useEffect(() => {
    checkBackend();
  }, []);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageUrl(e.target.result);
        setImage(file);
        setResults(null);
        setError('');
      };
      reader.readAsDataURL(file);
    }
  };

  const detectObjects = async () => {
    if (!imageUrl) return;
    
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('http://localhost:5000/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json', 
        },
        body: JSON.stringify({ image: imageUrl })
      });

      if (!response.ok) {
        throw new Error('Backend server error');
      }

      const detections = await response.json();
      setResults(detections);
      setBackendStatus('online');
    } catch (err) {
      setError('Detection failed. Make sure the Flask backend is running on http://localhost:5000');
      setBackendStatus('offline');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const drawBoundingBoxes = () => {
    if (!results || !imageUrl) return null;

    return (
      <div className="relative inline-block">
        <img src={imageUrl} alt="Analyzed" className="max-w-full h-auto rounded-lg" />
        <svg className="absolute top-0 left-0 w-full h-full pointer-events-none">
          {results.map((detection, idx) => {
            const [x1, y1, x2, y2] = detection.bbox;
            return (
              <g key={idx}>
                <rect
                  x={`${x1}%`}
                  y={`${y1}%`}
                  width={`${x2 - x1}%`}
                  height={`${y2 - y1}%`}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="3"
                />
                <rect
                  x={`${x1}%`}
                  y={`${Math.max(0, y1 - 3)}%`}
                  width={`${Math.min(100 - x1, (detection.object.length + 5) * 0.8)}%`}
                  height="3%"
                  fill="#ef4444"
                />
                <text
                  x={`${x1 + 0.5}%`}
                  y={`${Math.max(2, y1 - 0.5)}%`}
                  fill="white"
                  fontSize="14"
                  fontWeight="bold"
                >
                  {detection.object} {detection.confidence}%
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-3">
            <Camera className="w-10 h-10 text-indigo-600" />
            <h1 className="text-4xl font-bold text-gray-800">YOLOS Object Detector</h1>
          </div>
          <p className="text-gray-600">Powered by HuggingFace YOLOS-Tiny Model</p>
          
          {/* Backend Status */}
          <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white shadow-sm">
            {backendStatus === 'online' ? (
              <>
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-green-700 font-medium">Backend Connected</span>
              </>
            ) : backendStatus === 'offline' ? (
              <>
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                <span className="text-sm text-red-700 font-medium">Backend Offline</span>
              </>
            ) : (
              <>
                <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                <span className="text-sm text-gray-600 font-medium">Checking...</span>
              </>
            )}
          </div>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          <div className="border-2 border-dashed border-indigo-300 rounded-xl p-12 text-center hover:border-indigo-500 transition-colors">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="imageUpload"
            />
            <label htmlFor="imageUpload" className="cursor-pointer">
              <Upload className="w-16 h-16 text-indigo-400 mx-auto mb-4" />
              <p className="text-lg font-semibold text-gray-700 mb-2">
                Click to upload an image
              </p>
              <p className="text-sm text-gray-500">
                PNG, JPG, WEBP, or any image format
              </p>
            </label>
          </div>

          {imageUrl && (
            <div className="mt-6">
              <button
                onClick={detectObjects}
                disabled={loading || backendStatus === 'offline'}
                className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Detecting Objects...
                  </>
                ) : (
                  <>
                    <Camera className="w-5 h-5" />
                    Detect Objects
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-red-800 font-medium">{error}</p>
              {backendStatus === 'offline' && (
                <p className="text-red-700 text-sm mt-2">
                  Run: <code className="bg-red-100 px-2 py-1 rounded">python flask_backend.py</code>
                </p>
              )}
            </div>
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-2 mb-6">
              <CheckCircle className="w-6 h-6 text-green-600" />
              <h2 className="text-2xl font-bold text-gray-800">
                Found {results.length} Object{results.length !== 1 ? 's' : ''}
              </h2>
            </div>
            
            {/* Image with bounding boxes */}
            <div className="mb-6 flex justify-center bg-gray-50 rounded-lg p-4">
              {drawBoundingBoxes()}
            </div>

            {/* Detection list */}
            <div className="space-y-3">
              {results.map((detection, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 border border-indigo-100"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                      {idx + 1}
                    </div>
                    <span className="font-semibold text-gray-800 text-lg">
                      {detection.object}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-600">Confidence</div>
                    <div className="font-bold text-indigo-600 text-xl">
                      {detection.confidence}%
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {results.length === 0 && (
              <div className="text-center py-12">
                <AlertCircle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500 text-lg">
                  No objects detected in this image
                </p>
                <p className="text-gray-400 text-sm mt-2">
                  Try adjusting the threshold in flask_backend.py
                </p>
              </div>
            )}
          </div>
        )}

        {/* Setup Instructions */}
        <div className="mt-8 bg-white rounded-lg shadow-md p-6 border-l-4 border-indigo-600">
          <h3 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
            <span className="text-2xl">🚀</span> Setup Instructions
          </h3>
          <div className="space-y-2 text-sm text-gray-700">
            <p><strong>1. Install dependencies:</strong></p>
            <code className="block bg-gray-100 px-3 py-2 rounded mt-1 mb-3">
              pip install flask flask-cors transformers torch pillow
            </code>
            
            <p><strong>2. Start the backend:</strong></p>
            <code className="block bg-gray-100 px-3 py-2 rounded mt-1 mb-3">
              python flask_backend.py
            </code>
            
            <p><strong>3. Upload an image and click "Detect Objects"</strong></p>
          </div>
        </div>
      </div>
    </div>
  );
}
