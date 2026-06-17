using System;
using System.Collections;
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
        public bool stretchPreviewToParent = true;

        WebCamTexture _webCamTexture;
        Color32[] _pixelBuffer;
        CameraFrame _latestFrame;
        bool _isRunning;
        bool _permissionGranted;
        int _frameCount;
        string _selectedDeviceName = string.Empty;
        string _lastStatus = "Camera not started";
        string _lastError = string.Empty;

        public Texture PreviewTexture => _webCamTexture;
        public bool IsRunning => _isRunning && _webCamTexture != null && _webCamTexture.isPlaying;
        public int FrameWidth => _webCamTexture != null ? _webCamTexture.width : requestedWidth;
        public int FrameHeight => _webCamTexture != null ? _webCamTexture.height : requestedHeight;
        public bool PermissionGranted => _permissionGranted;
        public bool HasReceivedFrame => _latestFrame != null && _latestFrame.pixels != null && _latestFrame.pixels.Length > 0;
        public int FrameCount => _frameCount;
        public string SelectedDeviceName => _selectedDeviceName;
        public string LastStatus => _lastStatus;
        public string LastError => _lastError;

        void Awake()
        {
            EnsurePreviewLayout();
        }

        void Start()
        {
            StartCoroutine(StartCameraWhenPermissionReady());
        }

        void OnDestroy()
        {
            StopCamera();
        }

        IEnumerator StartCameraWhenPermissionReady()
        {
            _lastError = string.Empty;
            _lastStatus = "Requesting WebCam permission...";

            if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
            {
                yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
            }

            _permissionGranted = Application.HasUserAuthorization(UserAuthorization.WebCam);
            if (!_permissionGranted)
            {
                _lastStatus = "WebCam permission denied";
                _lastError = "WebCam permission denied. Enable WebCam capability and allow camera permission on HoloLens.";
                Debug.LogError(_lastError);
                yield break;
            }

            StartCamera();
        }

        void Update()
        {
            if (!IsRunning)
                return;

            if (!_webCamTexture.didUpdateThisFrame)
                return;

            int width = _webCamTexture.width;
            int height = _webCamTexture.height;
            if (width <= 16 || height <= 16)
            {
                _lastStatus = "Camera is starting: " + width + "x" + height;
                return;
            }

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

            _frameCount++;
            _lastStatus = "Frame received: " + width + "x" + height + ", count=" + _frameCount;

            if (aspectFitter != null)
                aspectFitter.aspectRatio = width / (float)height;
        }

        public void StartCamera()
        {
            if (_isRunning)
                return;

            try
            {
                EnsurePreviewLayout();

                WebCamDevice[] devices = WebCamTexture.devices;
                WebCamDevice? selected = null;
                _selectedDeviceName = string.Empty;

                foreach (var device in devices)
                {
                    Debug.Log("WebCam device found: " + device.name + ", frontFacing=" + device.isFrontFacing);
                    if (!selected.HasValue && device.isFrontFacing == useFrontFacingCamera)
                        selected = device;
                }

                if (selected.HasValue)
                {
                    _selectedDeviceName = selected.Value.name;
                    _webCamTexture = new WebCamTexture(selected.Value.name, requestedWidth, requestedHeight, requestedFps);
                }
                else
                {
                    _selectedDeviceName = devices.Length > 0 ? devices[0].name : "Default WebCamTexture";
                    _webCamTexture = devices.Length > 0
                        ? new WebCamTexture(devices[0].name, requestedWidth, requestedHeight, requestedFps)
                        : new WebCamTexture(requestedWidth, requestedHeight, requestedFps);
                }

                if (previewImage != null)
                    previewImage.texture = _webCamTexture;

                _webCamTexture.Play();
                _isRunning = true;
                _lastError = string.Empty;
                _lastStatus = "Camera started: " + _selectedDeviceName + ", devices=" + devices.Length;
                Debug.Log(_lastStatus);
            }
            catch (Exception ex)
            {
                _isRunning = false;
                _lastStatus = "Camera start failed";
                _lastError = ex.GetType().Name + ": " + ex.Message;
                Debug.LogException(ex);
            }
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
            _lastStatus = "Camera stopped";
        }

        public bool TryGetLatestFrame(out CameraFrame frame)
        {
            frame = _latestFrame;
            return frame != null && frame.pixels != null && frame.pixels.Length > 0;
        }

        void EnsurePreviewLayout()
        {
            if (!stretchPreviewToParent || previewImage == null)
                return;

            RectTransform rect = previewImage.rectTransform;
            if (rect == null)
                return;

            bool looksZeroSized = rect.rect.width < 1f || rect.rect.height < 1f || rect.sizeDelta.sqrMagnitude < 1f;
            if (!looksZeroSized)
                return;

            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            rect.pivot = new Vector2(0.5f, 0.5f);
            rect.anchoredPosition = Vector2.zero;
            rect.sizeDelta = Vector2.zero;
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;
        }
    }
}
