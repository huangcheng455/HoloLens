using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

#if (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
using UnityEngine.XR.WSA.Input;
#endif

namespace HoloFaceRecognition
{
    [Serializable]
    public sealed class RecognizedFace
    {
        public Rect pixelRect;
        public string name;
        public float similarity;
        public float detectionConfidence;
    }

    public sealed class FaceRecognitionManager : MonoBehaviour
    {
        [Header("Scene References")]
        public HoloLensCameraReader cameraReader;
        public FaceOverlayRenderer overlayRenderer;
        public Text statsText;
        public InputField registerNameInput;
        public Text statusText;
        public Button clearDatabaseButton;

        [Header("Model")]
        public string modelFileName = "ghostfacenet.onnx";
        public string modelName = "GhostFaceNet";

        [Header("Runtime")]
        [Range(1, 30)] public int detectEveryNFrames = 5;
        [Range(0f, 1f)] public float threshold = 0.5f;
        public float bboxPadding = 0.25f;

        [Header("Debug")]
        public bool enablePipelineDebugMode = true;
        public bool logPipelineSteps = true;

        [Header("HoloLens Test")]
        public bool enableAirTapRegister = true;
        public string hololensDefaultRegisterName = "HoloLensUser";
        public bool createLegacyClearDatabaseButton;

        readonly FaceAligner _aligner = new FaceAligner();
        readonly FaceMatcher _matcher = new FaceMatcher();
        readonly FaceDetectorService _detector = new FaceDetectorService();
        readonly OnnxFaceRecognizer _recognizer = new OnnxFaceRecognizer();
        readonly FaceDatabase _database = new FaceDatabase();
        readonly List<RecognizedFace> _latestResults = new List<RecognizedFace>();

        bool _initialized;
        bool _detectorInitialized;
        bool _recognizerInitialized;
        bool _processing;
        bool _pendingAirTapRegister;
        int _frameCounter;
        float _fpsTimer;
        int _fpsFrames;
        float _fps;
        float _lastPipelineMs;
        float _lastCameraFrameMs;
        float _lastDetectorMs;
        float _lastCropMs;
        float _lastEmbeddingMs;
        float _lastDbMatchMs;
        float _lastRegisterDbMs;
        float[] _lastEmbedding;
        string _initStatus = "Not initialized";
        string _onnxStatus = "ONNX: NOT STARTED";
        string _lastPipelineStage = "Pipeline: idle";
        string _lastMatchStatus = "Match: not run";
        float _onnxInitMs;
        string _lastError = string.Empty;
        string _lastInteractionStatus = string.Empty;
        bool _lastRegisterSucceeded;
        GameObject _popupRoot;
        Text _popupText;
        Coroutine _popupRoutine;

#if (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
        GestureRecognizer _tapRecognizer;
#endif

        void Awake()
        {
            Screen.sleepTimeout = SleepTimeout.NeverSleep;
            EnsureDebugTextReadable();
            EnsureRegistrationUi();
            InitializeAirTapRegister();
        }

        async void Start()
        {
            try
            {
                await InitializeAsync();
            }
            catch (Exception ex)
            {
                _initialized = false;
                _initStatus = "Init failed";
                _lastError = ex.GetType().Name + ": " + ex.Message;
                UnityEngine.Debug.LogException(ex);
                SetStatusText(_lastError);
            }
        }

        void OnDestroy()
        {
            Screen.sleepTimeout = SleepTimeout.SystemSetting;
            DisposeAirTapRegister();
            _recognizer.Dispose();
        }

        void Update()
        {
            _matcher.Threshold = threshold;
            UpdateFps();

#if UNITY_EDITOR
            if (Input.GetKeyDown(KeyCode.R))
                RegisterCurrentFaceForHoloLensTest();
#endif

            if (_pendingAirTapRegister)
            {
                _pendingAirTapRegister = false;
                RegisterCurrentFaceForHoloLensTest();
            }

            if (!_initialized || !_detectorInitialized || _processing || cameraReader == null)
            {
                UpdateStatsText();
                return;
            }

            var cameraFrameSw = Stopwatch.StartNew();
            bool hasFrame = cameraReader.TryGetLatestFrame(out var frame);
            cameraFrameSw.Stop();
            _lastCameraFrameMs = (float)cameraFrameSw.Elapsed.TotalMilliseconds;
            if (!hasFrame)
            {
                _lastPipelineStage = "CameraFrame: waiting";
                UpdateStatsText();
                return;
            }

            _frameCounter++;
            if (_frameCounter % Mathf.Max(1, detectEveryNFrames) == 0)
                _ = ProcessFrameAsync(frame);

            UpdateStatsText();
        }

        async Task InitializeAsync()
        {
            _initStatus = "Loading database...";
            await _database.LoadAsync();

            _initStatus = "Initializing detector...";
            await _detector.InitializeAsync();
            _detectorInitialized = true;

            _initStatus = "Initializing recognizer...";
            _onnxStatus = "ONNX: INITIALIZING...";
            _onnxInitMs = 0f;
            var recognizerInitSw = Stopwatch.StartNew();
            try
            {
#if USE_ONNXRUNTIME
                string modelPath = await PrepareModelFileAsync(modelFileName);
                await _recognizer.InitializeAsync(modelPath, modelName);
                recognizerInitSw.Stop();
                _onnxInitMs = (float)recognizerInitSw.Elapsed.TotalMilliseconds;
                _recognizerInitialized = true;
                _onnxStatus = "Recognizer: READY";
                UnityEngine.Debug.Log("Recognizer: READY. ONNX init time: " + _onnxInitMs.ToString("0.0") + " ms");
#else
                recognizerInitSw.Stop();
                _onnxInitMs = (float)recognizerInitSw.Elapsed.TotalMilliseconds;
                _recognizerInitialized = false;
                _onnxStatus = "ONNX: DISABLED (mock mode)";
                UnityEngine.Debug.LogWarning("ONNX: DISABLED (mock mode). USE_ONNXRUNTIME is not defined; recognition is disabled to avoid silent mock embeddings.");
#endif
            }
            catch (Exception ex)
            {
                recognizerInitSw.Stop();
                _onnxInitMs = (float)recognizerInitSw.Elapsed.TotalMilliseconds;
                _recognizerInitialized = false;
                _onnxStatus = "ONNX: FAILED - " + BuildOnnxInitError(ex);
                _lastError = _onnxStatus;
                UnityEngine.Debug.LogWarning(_lastError);
                UnityEngine.Debug.LogException(ex);
            }

            _initialized = true;
            _initStatus = _recognizerInitialized ? "Initialized" : "Detection only";
        }

        static string BuildOnnxInitError(Exception ex)
        {
            if (ex is DllNotFoundException)
                return "DLL missing: " + ex.Message;

            if (ex is FileNotFoundException fileNotFound)
                return "model not found: " + (string.IsNullOrEmpty(fileNotFound.FileName) ? fileNotFound.Message : fileNotFound.FileName);

            if (ex is BadImageFormatException)
                return "DLL architecture mismatch: " + ex.Message;

            if (ex is TypeInitializationException)
                return "runtime initialization failed: " + ex.Message;

            return ex.GetType().Name + ": " + ex.Message;
        }

        static async Task<string> PrepareModelFileAsync(string fileName)
        {
            string runtimeDir = Path.Combine(Application.persistentDataPath, "Models");
            Directory.CreateDirectory(runtimeDir);

            string runtimePath = Path.Combine(runtimeDir, fileName);
            if (File.Exists(runtimePath))
                return runtimePath;

            string source = Application.streamingAssetsPath.TrimEnd('/', '\\') + "/" + fileName;
            if (source.Contains("://") || source.Contains(":///"))
            {
                using (UnityWebRequest request = UnityWebRequest.Get(source))
                {
                    var op = request.SendWebRequest();
                    while (!op.isDone)
                        await Task.Yield();

#if UNITY_2020_2_OR_NEWER
                    if (request.result != UnityWebRequest.Result.Success)
#else
                    if (request.isNetworkError || request.isHttpError)
#endif
                        throw new FileNotFoundException("Unable to read ONNX model from StreamingAssets.", source);

                    File.WriteAllBytes(runtimePath, request.downloadHandler.data);
                }
            }
            else
            {
                if (!File.Exists(source))
                    return source;

                File.Copy(source, runtimePath, true);
            }

            return runtimePath;
        }

        async Task ProcessFrameAsync(CameraFrame frame)
        {
            _processing = true;
            var sw = Stopwatch.StartNew();
            _lastDetectorMs = 0f;
            _lastCropMs = 0f;
            _lastEmbeddingMs = 0f;
            _lastDbMatchMs = 0f;
            _lastPipelineStage = "CameraFrame: received";
            PipelineLog("CameraFrame received: " + frame.width + "x" + frame.height + ", rotation=" + frame.rotationAngle + ", mirror=" + frame.verticallyMirrored);

            try
            {
                bool pipelineHadError = false;
                List<FaceDetectionResult> detections;
                var detectorSw = Stopwatch.StartNew();
                try
                {
                    _lastPipelineStage = "Detector: running";
                    detections = await _detector.DetectAsync(frame);
                    detectorSw.Stop();
                    _lastDetectorMs = (float)detectorSw.Elapsed.TotalMilliseconds;
                    _lastPipelineStage = "Detector: ok";
                    PipelineLog("Detector OK: faces=" + detections.Count + ", time=" + _lastDetectorMs.ToString("0.0") + " ms");
                }
                catch (Exception ex)
                {
                    detectorSw.Stop();
                    _lastDetectorMs = (float)detectorSw.Elapsed.TotalMilliseconds;
                    _lastError = "Detector failed: " + ex.GetType().Name + ": " + ex.Message;
                    _lastPipelineStage = "Detector: failed";
                    UnityEngine.Debug.LogError(_lastError);
                    throw;
                }

                var recognized = new List<RecognizedFace>();
                if (detections.Count == 0)
                {
                    _lastEmbedding = null;
                    _lastMatchStatus = "Match: skipped, no face";
                    PipelineLog("Pipeline stopped: no face detected.");
                }

                foreach (var detection in detections)
                {
                    var recognizedFace = new RecognizedFace
                    {
                        pixelRect = detection.pixelRect,
                        name = _recognizerInitialized ? "Recognizing..." : "Face",
                        similarity = 0f,
                        detectionConfidence = detection.confidence
                    };

                    if (_recognizerInitialized)
                    {
                        Color32[] aligned = null;
                        try
                        {
                            var cropSw = Stopwatch.StartNew();
                            _lastPipelineStage = "Crop: running";
                            aligned = _aligner.Align(frame, detection, bboxPadding);
                            cropSw.Stop();
                            _lastCropMs += (float)cropSw.Elapsed.TotalMilliseconds;
                            if (aligned == null || aligned.Length == 0)
                                throw new InvalidOperationException("Crop produced empty pixels.");
                            PipelineLog("Crop OK: pixels=" + aligned.Length + ", time=" + cropSw.Elapsed.TotalMilliseconds.ToString("0.0") + " ms");
                        }
                        catch (Exception ex)
                        {
                            _lastEmbedding = null;
                            recognizedFace.name = "Crop failed";
                            recognizedFace.similarity = 0f;
                            _lastError = "Crop failed: " + ex.GetType().Name + ": " + ex.Message;
                            _lastPipelineStage = "Crop: failed";
                            pipelineHadError = true;
                            UnityEngine.Debug.LogError(_lastError);
                            UnityEngine.Debug.LogException(ex);
                            recognized.Add(recognizedFace);
                            continue;
                        }

                        try
                        {
                            var embeddingSw = Stopwatch.StartNew();
                            _lastPipelineStage = "ONNX embedding: running";
                            float[] embedding = await _recognizer.ExtractEmbeddingAsync(aligned);
                            embeddingSw.Stop();
                            _lastEmbeddingMs += (float)embeddingSw.Elapsed.TotalMilliseconds;
                            if (embedding == null || embedding.Length == 0)
                                throw new InvalidOperationException("Recognizer produced empty embedding.");
                            _lastEmbedding = embedding;
                            PipelineLog("ONNX embedding OK: dim=" + embedding.Length + ", time=" + embeddingSw.Elapsed.TotalMilliseconds.ToString("0.0") + " ms");

                        }
                        catch (Exception ex)
                        {
                            _recognizerInitialized = false;
                            _lastEmbedding = null;
                            recognizedFace.name = "ONNX failed";
                            recognizedFace.similarity = 0f;
                            _lastError = "ONNX embedding failed: " + ex.GetType().Name + ": " + ex.Message;
                            _onnxStatus = "ONNX: FAILED - " + ex.GetType().Name + ": " + ex.Message;
                            _lastPipelineStage = "ONNX embedding: failed";
                            pipelineHadError = true;
                            UnityEngine.Debug.LogError(_lastError);
                            UnityEngine.Debug.LogException(ex);
                        }

                        if (_lastEmbedding != null)
                        {
                            try
                            {
                                var matchSw = Stopwatch.StartNew();
                                _lastPipelineStage = "DB match: running";
                                FaceMatchResult match = _matcher.Match(_lastEmbedding, _database);
                                matchSw.Stop();
                                _lastDbMatchMs += (float)matchSw.Elapsed.TotalMilliseconds;
                                recognizedFace.name = match.name;
                                recognizedFace.similarity = match.similarity;
                                _lastMatchStatus = string.Format("Match: {0} sim={1:0.000}", match.name, match.similarity);
                                _lastPipelineStage = "Match: ok";
                                PipelineLog("DB match OK: name=" + match.name + ", similarity=" + match.similarity.ToString("0.000") + ", known=" + match.isKnown + ", time=" + matchSw.Elapsed.TotalMilliseconds.ToString("0.0") + " ms");
                            }
                            catch (Exception ex)
                            {
                                _lastEmbedding = null;
                                recognizedFace.name = "Match failed";
                                recognizedFace.similarity = 0f;
                                _lastError = "DB match failed: " + ex.GetType().Name + ": " + ex.Message;
                                _lastMatchStatus = "Match: failed";
                                _lastPipelineStage = "DB match: failed";
                                pipelineHadError = true;
                                UnityEngine.Debug.LogError(_lastError);
                                UnityEngine.Debug.LogException(ex);
                            }
                        }
                    }
                    else
                    {
                        _lastEmbedding = null;
                        _lastMatchStatus = "Match: skipped, ONNX unavailable";
                        PipelineLog("ONNX embedding skipped: recognizer is not initialized.");
                    }

                    recognized.Add(recognizedFace);
                }

                _latestResults.Clear();
                _latestResults.AddRange(recognized);

                if (overlayRenderer != null)
                    overlayRenderer.SetFaces(_latestResults, frame);

                if (_recognizerInitialized && !pipelineHadError)
                    _lastError = string.Empty;
            }
            catch (Exception ex)
            {
                _lastError = "Pipeline " + ex.GetType().Name + ": " + ex.Message;
                UnityEngine.Debug.LogException(ex);
            }
            finally
            {
                _lastPipelineMs = (float)sw.Elapsed.TotalMilliseconds;
                _processing = false;
            }
        }

        public void RegisterCurrentFace(string personName)
        {
            _lastRegisterSucceeded = false;
            personName = personName == null ? string.Empty : personName.Trim();
            if (string.IsNullOrEmpty(personName))
            {
                UnityEngine.Debug.LogWarning("Cannot register face: name is empty.");
                SetStatusText("Cannot register: name is empty");
                return;
            }

            if (_lastEmbedding == null)
            {
                string message = _recognizerInitialized
                    ? "No face embedding available"
                    : "Detection works, but recognition is unavailable. Add UWP ARM64 ONNX Runtime to register faces.";
                UnityEngine.Debug.LogWarning(message);
                SetStatusText(message);
                return;
            }

            var registerSw = Stopwatch.StartNew();
            try
            {
                _database.AddEmbedding(personName, _lastEmbedding, _recognizer.ModelName);
                _database.Save();
                registerSw.Stop();
                _lastRegisterDbMs = (float)registerSw.Elapsed.TotalMilliseconds;
                _lastRegisterSucceeded = true;
                _lastInteractionStatus = "Registered " + personName + " (" + _lastRegisterDbMs.ToString("0.0") + " ms)";
                PipelineLog("DB register OK: name=" + personName + ", embeddings=" + GetEmbeddingCount() + ", time=" + _lastRegisterDbMs.ToString("0.0") + " ms");
            }
            catch (Exception ex)
            {
                registerSw.Stop();
                _lastRegisterDbMs = (float)registerSw.Elapsed.TotalMilliseconds;
                _lastError = "DB register failed: " + ex.GetType().Name + ": " + ex.Message;
                _lastInteractionStatus = "Register failed at DB";
                UnityEngine.Debug.LogError(_lastError);
                UnityEngine.Debug.LogException(ex);
                SetStatusText(_lastError);
                throw;
            }
        }

        public void RegisterCurrentFaceFromUI()
        {
            string personName = registerNameInput == null ? string.Empty : registerNameInput.text.Trim();
            if (string.IsNullOrEmpty(personName))
            {
                UnityEngine.Debug.LogWarning("Please enter a name.");
                SetStatusText("Please enter a name");
                return;
            }

            if (_latestResults.Count == 0 || _lastEmbedding == null)
            {
                string message = _latestResults.Count == 0
                    ? "No face available"
                    : "Face detected, but recognition is unavailable. Add UWP ARM64 ONNX Runtime to register.";
                UnityEngine.Debug.LogWarning(message);
                SetStatusText(message);
                return;
            }

            try
            {
                RegisterCurrentFace(personName);
                if (_lastRegisterSucceeded)
                {
                    SetStatusText("Registered: " + personName);
                    ShowPopup("Registered: " + personName);
                }
            }
            catch (Exception)
            {
                ShowPopup("Register failed: DB error");
            }
        }

        public void RegisterCurrentFaceForHoloLensTest()
        {
            string personName = registerNameInput == null ? string.Empty : registerNameInput.text.Trim();
            if (string.IsNullOrEmpty(personName))
                personName = string.IsNullOrWhiteSpace(hololensDefaultRegisterName) ? "HoloLensUser" : hololensDefaultRegisterName.Trim();

            _lastInteractionStatus = "Air tap register triggered";

            if (_latestResults.Count == 0 || _lastEmbedding == null)
            {
                string message = _latestResults.Count == 0
                    ? "Air tap: no face available"
                    : "Face detected, recognition unavailable";
                SetStatusText(message);
                ShowPopup(message);
                return;
            }

            try
            {
                RegisterCurrentFace(personName);
                if (_lastRegisterSucceeded)
                {
                    SetStatusText("Air tap registered: " + personName);
                    ShowPopup("Registered: " + personName);
                }
            }
            catch (Exception)
            {
                ShowPopup("Register failed: DB error");
            }
        }

        public void ClearFaceDatabaseFromUI()
        {
            _database.Clear();
            _database.Save();
            SetStatusText("Ready");
            ShowPopup("Face database cleared");
        }

        void SetStatusText(string message)
        {
            if (statusText != null)
            {
                statusText.text = message;
                return;
            }

            if (statsText != null)
                statsText.text = message;
        }

        void EnsureRegistrationUi()
        {
            if (registerNameInput == null)
            {
                GameObject inputObject = GameObject.Find("RegisterNameInput");
                if (inputObject != null)
                    registerNameInput = inputObject.GetComponent<InputField>();
            }

            if (statusText == null)
            {
                GameObject statusObject = GameObject.Find("RegisterStatusText");
                if (statusObject != null)
                    statusText = statusObject.GetComponent<Text>();
            }

            Button registerButton = FindButton("RegisterButton");
            bool createdClearButton = false;
            if (clearDatabaseButton == null)
                clearDatabaseButton = FindButton("ClearDatabaseButton");

            if (clearDatabaseButton == null && createLegacyClearDatabaseButton)
            {
                clearDatabaseButton = CreateClearDatabaseButton(registerButton);
                createdClearButton = clearDatabaseButton != null;
            }

            if (clearDatabaseButton != null && (createdClearButton || clearDatabaseButton.onClick.GetPersistentEventCount() == 0))
                clearDatabaseButton.onClick.AddListener(ClearFaceDatabaseFromUI);
        }

        static Button FindButton(string objectName)
        {
            GameObject buttonObject = GameObject.Find(objectName);
            return buttonObject == null ? null : buttonObject.GetComponent<Button>();
        }

        Button CreateClearDatabaseButton(Button registerButton)
        {
            Canvas canvas = FindObjectOfType<Canvas>();
            if (canvas == null)
                return null;

            Transform parent = registerButton != null ? registerButton.transform.parent : canvas.transform;
            var buttonObject = new GameObject("ClearDatabaseButton", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Button));
            buttonObject.layer = canvas.gameObject.layer;
            buttonObject.transform.SetParent(parent, false);

            RectTransform rect = buttonObject.GetComponent<RectTransform>();
            RectTransform registerRect = registerButton == null ? null : registerButton.GetComponent<RectTransform>();
            if (registerRect != null)
            {
                rect.anchorMin = registerRect.anchorMin;
                rect.anchorMax = registerRect.anchorMax;
                rect.pivot = registerRect.pivot;
                rect.sizeDelta = registerRect.sizeDelta;
                rect.anchoredPosition = registerRect.anchoredPosition + new Vector2(0f, 70f);
            }
            else
            {
                rect.anchorMin = new Vector2(0f, 0f);
                rect.anchorMax = new Vector2(0f, 0f);
                rect.pivot = new Vector2(0f, 0f);
                rect.sizeDelta = new Vector2(180f, 60f);
                rect.anchoredPosition = new Vector2(20f, 90f);
            }

            Image image = buttonObject.GetComponent<Image>();
            image.color = Color.white;

            Button button = buttonObject.GetComponent<Button>();
            button.targetGraphic = image;

            Text label = CreateText(buttonObject.transform, "Text", "Clear DB", 24, TextAnchor.MiddleCenter, new Color(0.196f, 0.196f, 0.196f, 1f));
            RectTransform labelRect = label.rectTransform;
            labelRect.anchorMin = Vector2.zero;
            labelRect.anchorMax = Vector2.one;
            labelRect.offsetMin = Vector2.zero;
            labelRect.offsetMax = Vector2.zero;

            return button;
        }

        void ShowPopup(string message)
        {
            EnsurePopup();
            if (_popupRoot == null || _popupText == null)
                return;

            _popupText.text = message;
            _popupRoot.SetActive(true);

            if (_popupRoutine != null)
                StopCoroutine(_popupRoutine);

            _popupRoutine = StartCoroutine(HidePopupAfterDelay(2f));
        }

        IEnumerator HidePopupAfterDelay(float delaySeconds)
        {
            yield return new WaitForSeconds(delaySeconds);
            if (_popupRoot != null)
                _popupRoot.SetActive(false);
            _popupRoutine = null;
        }

        void EnsurePopup()
        {
            if (_popupRoot != null)
                return;

            Canvas canvas = FindObjectOfType<Canvas>();
            if (canvas == null)
                return;

            _popupRoot = new GameObject("RegisterPopup", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
            _popupRoot.layer = canvas.gameObject.layer;
            _popupRoot.transform.SetParent(canvas.transform, false);

            RectTransform rootRect = _popupRoot.GetComponent<RectTransform>();
            rootRect.anchorMin = new Vector2(0.5f, 0.5f);
            rootRect.anchorMax = new Vector2(0.5f, 0.5f);
            rootRect.pivot = new Vector2(0.5f, 0.5f);
            rootRect.sizeDelta = new Vector2(420f, 96f);
            rootRect.anchoredPosition = Vector2.zero;

            Image background = _popupRoot.GetComponent<Image>();
            background.color = new Color(0f, 0f, 0f, 0.78f);

            _popupText = CreateText(_popupRoot.transform, "Message", string.Empty, 28, TextAnchor.MiddleCenter, Color.white);
            RectTransform textRect = _popupText.rectTransform;
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.offsetMin = new Vector2(20f, 12f);
            textRect.offsetMax = new Vector2(-20f, -12f);

            _popupRoot.SetActive(false);
        }

        static Text CreateText(Transform parent, string objectName, string text, int fontSize, TextAnchor alignment, Color color)
        {
            var textObject = new GameObject(objectName, typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
            textObject.layer = parent.gameObject.layer;
            textObject.transform.SetParent(parent, false);

            Text label = textObject.GetComponent<Text>();
            label.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            label.fontSize = fontSize;
            label.alignment = alignment;
            label.color = color;
            label.text = text;
            return label;
        }

        void UpdateFps()
        {
            _fpsFrames++;
            _fpsTimer += Time.unscaledDeltaTime;
            if (_fpsTimer >= 1f)
            {
                _fps = _fpsFrames / _fpsTimer;
                _fpsFrames = 0;
                _fpsTimer = 0f;
            }
        }

        void UpdateStatsText()
        {
            if (statsText == null)
                return;

            string cameraLine = "Camera: missing";
            if (cameraReader != null)
            {
                cameraLine = string.Format(
                    "Camera: {0} {1}x{2} frames={3} perm={4} rotation={5} mirror={6} | {7}",
                    cameraReader.IsRunning ? "running" : "not running",
                    cameraReader.FrameWidth,
                    cameraReader.FrameHeight,
                    cameraReader.FrameCount,
                    cameraReader.PermissionGranted ? "yes" : "no",
                    cameraReader.RotationAngle,
                    cameraReader.VerticallyMirrored ? "vertical" : "none",
                    cameraReader.LastStatus);
            }

            string detectorStatus = _detector.LastStatus;
            string error = BuildErrorLine();
            string detectorLine = string.Format("Detector: {0} faces={1}", detectorStatus, _latestResults.Count);
            string onnxLine = string.Format("ONNX status: {0} init={1:0.0} ms", _onnxStatus, _onnxInitMs);
            if (!string.IsNullOrEmpty(error))
                onnxLine += " | " + error;
            string latencyLine = string.Format(
                "FPS: {0:0.0} latency={1:0.0} ms inference={2:0.0} ms DB={3}/{4}",
                _fps,
                _lastPipelineMs,
                _recognizer.LastInferenceMs,
                GetPersonCount(),
                GetEmbeddingCount());
            if (enablePipelineDebugMode)
            {
                latencyLine += string.Format(
                    " | {0} | cam={1:0.0} det={2:0.0} crop={3:0.0} onnx={4:0.0} match={5:0.0} regdb={6:0.0} ms | {7}",
                    _lastPipelineStage,
                    _lastCameraFrameMs,
                    _lastDetectorMs,
                    _lastCropMs,
                    _lastEmbeddingMs,
                    _lastDbMatchMs,
                    _lastRegisterDbMs,
                    _lastMatchStatus);
            }

            statsText.text = string.Format(
                "{0}\n{1}\n{2}\n{3}",
                cameraLine,
                detectorLine,
                onnxLine,
                string.IsNullOrEmpty(_lastInteractionStatus) ? latencyLine : latencyLine + " | " + _lastInteractionStatus);
        }

        string BuildErrorLine()
        {
            if (!string.IsNullOrEmpty(_lastError))
                return "Error: " + _lastError;

            if (cameraReader != null && !string.IsNullOrEmpty(cameraReader.LastError))
                return "Camera error: " + cameraReader.LastError;

            if (!string.IsNullOrEmpty(_detector.LastError))
                return "Detector error: " + _detector.LastError;

            return string.Empty;
        }

        void PipelineLog(string message)
        {
            if (enablePipelineDebugMode && logPipelineSteps)
                UnityEngine.Debug.Log("[FacePipeline] " + message);
        }

        int GetPersonCount()
        {
            return _database.Data != null && _database.Data.people != null ? _database.Data.people.Count : 0;
        }

        int GetEmbeddingCount()
        {
            int count = 0;
            if (_database.Data == null || _database.Data.people == null)
                return 0;

            foreach (var person in _database.Data.people)
            {
                if (person != null && person.embeddings != null)
                    count += person.embeddings.Count;
            }

            return count;
        }

        void EnsureDebugTextReadable()
        {
            if (statsText == null)
                return;

            statsText.fontSize = Mathf.Max(statsText.fontSize, 28);
            statsText.color = Color.white;
            statsText.alignment = TextAnchor.UpperLeft;
            statsText.horizontalOverflow = HorizontalWrapMode.Overflow;
            statsText.verticalOverflow = VerticalWrapMode.Overflow;
            EnsureStatsBackgroundPanel();

            RectTransform rect = statsText.rectTransform;
            if (rect != null)
                rect.sizeDelta = new Vector2(Mathf.Max(rect.sizeDelta.x, 1040f), Mathf.Max(rect.sizeDelta.y, 180f));
        }

        void EnsureStatsBackgroundPanel()
        {
            RectTransform textRect = statsText.rectTransform;
            if (textRect == null || textRect.parent == null)
                return;

            const string panelName = "StatsTextBackgroundPanel";
            Transform existing = textRect.parent.Find(panelName);
            RectTransform panelRect;
            Image panelImage;

            if (existing == null)
            {
                var panel = new GameObject(panelName, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
                panel.layer = statsText.gameObject.layer;
                panel.transform.SetParent(textRect.parent, false);
                panel.transform.SetSiblingIndex(textRect.GetSiblingIndex());
                panelRect = panel.GetComponent<RectTransform>();
                panelImage = panel.GetComponent<Image>();
            }
            else
            {
                panelRect = existing as RectTransform;
                panelImage = existing.GetComponent<Image>();
            }

            if (panelRect == null || panelImage == null)
                return;

            panelRect.anchorMin = textRect.anchorMin;
            panelRect.anchorMax = textRect.anchorMax;
            panelRect.pivot = textRect.pivot;
            panelRect.anchoredPosition = textRect.anchoredPosition;
            panelRect.sizeDelta = new Vector2(Mathf.Max(textRect.sizeDelta.x, 1040f), Mathf.Max(textRect.sizeDelta.y, 180f));
            panelImage.color = new Color(0f, 0f, 0f, 0.5f);
            panelImage.raycastTarget = false;
        }

        void InitializeAirTapRegister()
        {
#if (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
            if (!enableAirTapRegister)
                return;

            try
            {
                _tapRecognizer = new GestureRecognizer();
                _tapRecognizer.SetRecognizableGestures(GestureSettings.Tap);
                _tapRecognizer.Tapped += OnAirTapped;
                _tapRecognizer.StartCapturingGestures();
                _lastInteractionStatus = "Air tap register enabled";
            }
            catch (Exception ex)
            {
                _lastInteractionStatus = "Air tap init failed";
                _lastError = ex.GetType().Name + ": " + ex.Message;
                UnityEngine.Debug.LogException(ex);
            }
#endif
        }

        void DisposeAirTapRegister()
        {
#if (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
            if (_tapRecognizer == null)
                return;

            _tapRecognizer.Tapped -= OnAirTapped;
            _tapRecognizer.StopCapturingGestures();
            _tapRecognizer.Dispose();
            _tapRecognizer = null;
#endif
        }

#if (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
        void OnAirTapped(TappedEventArgs args)
        {
            _pendingAirTapRegister = true;
        }
#endif
    }
}
