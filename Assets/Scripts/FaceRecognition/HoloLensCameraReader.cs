using System;
using UnityEngine;
using UnityEngine.UI;

namespace HoloFaceRecognition
{
    [Serializable]
    public sealed class CameraFrame
    {
        public int width;
        public int height;
        public Color32[] pixels;
        public double timestamp;
    }

    public sealed class HoloLensCameraReader : MonoBehaviour
    {
        [Header("Camera")]
        public int requestedWidth = 640;
        public int requestedHeight = 360;
        public int requestedFps = 15;
        public bool useFrontFacingCamera;

        [Header("Preview")]
        public RawImage previewImage;
        public AspectRatioFitter aspectFitter;

        WebCamTexture _webCamTexture;
        Color32[] _pixelBuffer;
        CameraFrame _latestFrame;
        bool _isRunning;

        public Texture PreviewTexture => _webCamTexture;
        public bool IsRunning => _isRunning && _webCamTexture != null && _webCamTexture.isPlaying;
        public int FrameWidth => _webCamTexture != null ? _webCamTexture.width : requestedWidth;
        public int FrameHeight => _webCamTexture != null ? _webCamTexture.height : requestedHeight;

        void Start()
        {
            StartCamera();
        }

        void OnDestroy()
        {
            StopCamera();
        }

        void Update()
        {
            if (!IsRunning || !_webCamTexture.didUpdateThisFrame)
                return;

            int width = _webCamTexture.width;
            int height = _webCamTexture.height;
            if (width <= 16 || height <= 16)
                return;

            int pixelCount = width * height;
            if (_pixelBuffer == null || _pixelBuffer.Length != pixelCount)
                _pixelBuffer = new Color32[pixelCount];

            _webCamTexture.GetPixels32(_pixelBuffer);

            var copy = new Color32[pixelCount];
            Array.Copy(_pixelBuffer, copy, pixelCount);
            _latestFrame = new CameraFrame
            {
                width = width,
                height = height,
                pixels = copy,
                timestamp = Time.realtimeSinceStartup
            };

            if (aspectFitter != null)
                aspectFitter.aspectRatio = width / (float)height;
        }

        public void StartCamera()
        {
            if (_isRunning)
                return;

            WebCamDevice? selected = null;
            foreach (var device in WebCamTexture.devices)
            {
                if (device.isFrontFacing == useFrontFacingCamera)
                {
                    selected = device;
                    break;
                }
            }

            if (selected.HasValue)
                _webCamTexture = new WebCamTexture(selected.Value.name, requestedWidth, requestedHeight, requestedFps);
            else
                _webCamTexture = new WebCamTexture(requestedWidth, requestedHeight, requestedFps);
            _webCamTexture.Play();
            _isRunning = true;

            if (previewImage != null)
                previewImage.texture = _webCamTexture;
        }

        public void StopCamera()
        {
            _isRunning = false;
            if (_webCamTexture == null)
                return;

            if (_webCamTexture.isPlaying)
                _webCamTexture.Stop();

            Destroy(_webCamTexture);
            _webCamTexture = null;
        }

        public bool TryGetLatestFrame(out CameraFrame frame)
        {
            frame = _latestFrame;
            return frame != null && frame.pixels != null && frame.pixels.Length > 0;
        }
    }
}
