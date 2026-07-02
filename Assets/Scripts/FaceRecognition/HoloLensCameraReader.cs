using System;
using System.Collections;
using System.Collections.Generic;
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
        public int rotationAngle;
        public bool verticallyMirrored;
    }

    public sealed class HoloLensCameraReader : MonoBehaviour
    {
        [Header("Camera")]
        public int requestedWidth = 640;
        public int requestedHeight = 360;
        public int requestedFps = 30;
        [Range(1, 30)] public int captureEveryNFrames = 3;
        public bool requestPermissionOnStart;
        public bool autoRetryCamera = true;
        public float retryIntervalSeconds = 3f;
        public float firstFrameTimeoutSeconds = 6f;
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
        float _nextRetryTime;
        bool _startingCamera;
        float _cameraStartTime;
        int _webCamUpdateCount;
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
        public int RotationAngle => _latestFrame != null ? _latestFrame.rotationAngle : (_webCamTexture != null ? _webCamTexture.videoRotationAngle : 0);
        public bool VerticallyMirrored => _latestFrame != null ? _latestFrame.verticallyMirrored : (_webCamTexture != null && _webCamTexture.videoVerticallyMirrored);
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

            if (!Application.HasUserAuthorization(UserAuthorization.WebCam) && requestPermissionOnStart)
            {
                yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
            }

            _permissionGranted = Application.HasUserAuthorization(UserAuthorization.WebCam);
            if (!_permissionGranted)
            {
                _lastStatus = "WebCam permission denied";
                _lastError = "WebCam permission denied. Enable Camera permission for this app in HoloLens Settings > Privacy > Camera.";
                Debug.LogError(_lastError);
                yield break;
            }

            StartCamera();
        }

        void Update()
        {
            if (!IsRunning)
            {
                RetryCameraIfReady();
                return;
            }

            if (!_webCamTexture.didUpdateThisFrame)
            {
                RestartCameraIfFirstFrameTimedOut();
                return;
            }

            _webCamUpdateCount++;
            if (_webCamUpdateCount % Mathf.Max(1, captureEveryNFrames) != 0)
                return;

            int width = _webCamTexture.width;
            int height = _webCamTexture.height;
            if (width <= 16 || height <= 16)
            {
                _lastStatus = "Camera is starting: " + width + "x" + height;
                RestartCameraIfFirstFrameTimedOut();
                return;
            }

            int pixelCount = width * height;
            if (_pixelBuffer == null || _pixelBuffer.Length != pixelCount)
                _pixelBuffer = new Color32[pixelCount];

            _webCamTexture.GetPixels32(_pixelBuffer);
            int rotationAngle = _webCamTexture.videoRotationAngle;
            bool verticallyMirrored = _webCamTexture.videoVerticallyMirrored;

            var copy = new Color32[pixelCount];
            Array.Copy(_pixelBuffer, copy, pixelCount);
            _latestFrame = new CameraFrame
            {
                width = width,
                height = height,
                pixels = copy,
                timestamp = Time.realtimeSinceStartup,
                rotationAngle = rotationAngle,
                verticallyMirrored = verticallyMirrored
            };

            _frameCount++;
            _lastStatus = "Frame received: " + width + "x" + height + ", count=" + _frameCount;

            if (aspectFitter != null)
                aspectFitter.aspectRatio = width / (float)height;
        }

        public void StartCamera()
        {
            if (_isRunning || _startingCamera)
                return;

            try
            {
                _startingCamera = true;
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

                var candidateDeviceNames = new List<string>();
                if (selected.HasValue)
                    candidateDeviceNames.Add(selected.Value.name);

                foreach (var device in devices)
                {
                    if (!candidateDeviceNames.Contains(device.name))
                        candidateDeviceNames.Add(device.name);
                }

                if (candidateDeviceNames.Count == 0)
                    candidateDeviceNames.Add(string.Empty);

                Exception lastException = null;
                foreach (string deviceName in candidateDeviceNames)
                {
                    foreach (int[] mode in GetCameraModes())
                    {
                        if (TryStartWebCamTexture(deviceName, mode[0], mode[1], mode[2], devices.Length, out lastException))
                            return;
                    }
                }

                throw lastException ?? new InvalidOperationException("No WebCamTexture mode could be started.");
            }
            catch (Exception ex)
            {
                _isRunning = false;
                _lastStatus = "Camera start failed";
                _lastError = ex.GetType().Name + ": " + ex.Message;
                _nextRetryTime = Time.realtimeSinceStartup + Mathf.Max(1f, retryIntervalSeconds);
                Debug.LogException(ex);
            }
            finally
            {
                _startingCamera = false;
            }
        }

        void RetryCameraIfReady()
        {
            if (!autoRetryCamera || _startingCamera)
                return;

            if (Time.realtimeSinceStartup < _nextRetryTime)
                return;

            _permissionGranted = Application.HasUserAuthorization(UserAuthorization.WebCam);
            if (!_permissionGranted)
            {
                _lastStatus = "WebCam permission denied";
                _lastError = "WebCam permission denied. Enable Camera permission for this app in HoloLens Settings > Privacy > Camera.";
                _nextRetryTime = Time.realtimeSinceStartup + Mathf.Max(1f, retryIntervalSeconds);
                return;
            }

            _lastStatus = "Retrying camera start...";
            _nextRetryTime = Time.realtimeSinceStartup + Mathf.Max(1f, retryIntervalSeconds);
            StartCamera();
        }

        IEnumerable<int[]> GetCameraModes()
        {
            yield return new[] { requestedWidth, requestedHeight, requestedFps };
            yield return new[] { 896, 504, 30 };
            yield return new[] { 1280, 720, 30 };
            yield return new[] { 640, 480, 30 };
            yield return new[] { 640, 360, 30 };
            yield return new[] { 640, 360, 15 };
        }

        bool TryStartWebCamTexture(string deviceName, int width, int height, int fps, int deviceCount, out Exception exception)
        {
            exception = null;
            try
            {
                _selectedDeviceName = string.IsNullOrEmpty(deviceName) ? "Default WebCamTexture" : deviceName;
                Debug.Log("Trying WebCamTexture: " + _selectedDeviceName + ", " + width + "x" + height + "@" + fps);

                _webCamTexture = string.IsNullOrEmpty(deviceName)
                    ? new WebCamTexture(width, height, fps)
                    : new WebCamTexture(deviceName, width, height, fps);

                _webCamTexture.Play();

                if (previewImage != null)
                    previewImage.texture = _webCamTexture;

                _isRunning = true;
                _cameraStartTime = Time.realtimeSinceStartup;
                _webCamUpdateCount = 0;
                _lastError = string.Empty;
                _lastStatus = "Camera started: " + _selectedDeviceName + ", requested=" + width + "x" + height + "@" + fps + ", devices=" + deviceCount;
                Debug.Log(_lastStatus);
                return true;
            }
            catch (Exception ex)
            {
                exception = ex;
                Debug.LogWarning("WebCamTexture mode failed: " + _selectedDeviceName + ", " + width + "x" + height + "@" + fps + " - " + ex.GetType().Name + ": " + ex.Message);

                if (_webCamTexture != null)
                {
                    if (_webCamTexture.isPlaying)
                        _webCamTexture.Stop();

                    Destroy(_webCamTexture);
                    _webCamTexture = null;
                }

                return false;
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

        void RestartCameraIfFirstFrameTimedOut()
        {
            if (!autoRetryCamera || firstFrameTimeoutSeconds <= 0f || HasReceivedFrame)
                return;

            if (Time.realtimeSinceStartup - _cameraStartTime < firstFrameTimeoutSeconds)
                return;

            _lastStatus = "Camera start timed out before first frame; retrying";
            _lastError = "Camera produced no frame after " + firstFrameTimeoutSeconds.ToString("0.0") + " seconds. Check camera privacy, close other camera apps, or reboot HoloLens.";
            Debug.LogWarning(_lastError);

            StopCamera();
            _nextRetryTime = Time.realtimeSinceStartup + Mathf.Max(1f, retryIntervalSeconds);
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
