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

        [Header("HoloLens Test")]
        public bool enableAirTapRegister = true;
        public string hololensDefaultRegisterName = "HoloLensUser";

        readonly FaceAligner _aligner = new FaceAligner();
        readonly FaceMatcher _matcher = new FaceMatcher();
        readonly FaceDetectorService _detector = new FaceDetectorService();
        readonly OnnxFaceRecognizer _recognizer = new OnnxFaceRecognizer();
        readonly FaceDatabase _database = new FaceDatabase();
        readonly List<RecognizedFace> _latestResults = new List<RecognizedFace>();

        bool _initialized;
        bool _processing;
        bool _pendingAirTapRegister;
        int _frameCounter;
        float _fpsTimer;
        int _fpsFrames;
        float _fps;
        float _lastPipelineMs;
        float[] _lastEmbedding;
        string _initStatus = "Not initialized";
        string _lastError = string.Empty;
        string _lastInteractionStatus = string.Empty;
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

            if (!_initialized || _processing || cameraReader == null || !cameraReader.TryGetLatestFrame(out var frame))
            {
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

            _initStatus = "Initializing recognizer...";
#if USE_ONNXRUNTIME
            string modelPath = await PrepareModelFileAsync(modelFileName);
            await _recognizer.InitializeAsync(modelPath, modelName);
#else
            await _recognizer.InitializeAsync(modelFileName, modelName);
#endif

            _initialized = true;
            _initStatus = "Initialized";
            _lastError = string.Empty;
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

            try
            {
                List<FaceDetectionResult> detections = await _detector.DetectAsync(frame);
                var recognized = new List<RecognizedFace>();
                if (detections.Count == 0)
                    _lastEmbedding = null;

                foreach (var detection in detections)
                {
                    Color32[] aligned = _aligner.Align(frame, detection, bboxPadding);
                    float[] embedding = await _recognizer.ExtractEmbeddingAsync(aligned);
                    _lastEmbedding = embedding;

                    FaceMatchResult match = _matcher.Match(embedding, _database);
                    recognized.Add(new RecognizedFace
                    {
                        pixelRect = detection.pixelRect,
                        name = match.name,
                        similarity = match.similarity,
                        detectionConfidence = detection.confidence
                    });
                }

                _latestResults.Clear();
                _latestResults.AddRange(recognized);

                if (overlayRenderer != null)
                    overlayRenderer.SetFaces(_latestResults, frame.width, frame.height);

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
            personName = personName == null ? string.Empty : personName.Trim();
            if (string.IsNullOrEmpty(personName))
            {
                UnityEngine.Debug.LogWarning("Cannot register face: name is empty.");
                SetStatusText("Cannot register: name is empty");
                return;
            }

            if (_lastEmbedding == null)
            {
                UnityEngine.Debug.LogWarning("No face embedding is available yet.");
                SetStatusText("No face embedding available");
                return;
            }

            _database.AddEmbedding(personName, _lastEmbedding, _recognizer.ModelName);
            _database.Save();
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
                UnityEngine.Debug.LogWarning("No face available.");
                SetStatusText("No face available");
                return;
            }

            RegisterCurrentFace(personName);
            SetStatusText("Registered: " + personName);
            ShowPopup("Registered: " + personName);
        }

        public void RegisterCurrentFaceForHoloLensTest()
        {
            string personName = registerNameInput == null ? string.Empty : registerNameInput.text.Trim();
            if (string.IsNullOrEmpty(personName))
                personName = string.IsNullOrWhiteSpace(hololensDefaultRegisterName) ? "HoloLensUser" : hololensDefaultRegisterName.Trim();

            _lastInteractionStatus = "Air tap register triggered";

            if (_latestResults.Count == 0 || _lastEmbedding == null)
            {
                SetStatusText("Air tap: no face available");
                ShowPopup("No face available");
                return;
            }

            RegisterCurrentFace(personName);
            SetStatusText("Air tap registered: " + personName);
            ShowPopup("Registered: " + personName);
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

            if (clearDatabaseButton == null)
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
                    "Camera: {0} {1}x{2} frames={3} perm={4}",
                    cameraReader.IsRunning ? "running" : "not running",
                    cameraReader.FrameWidth,
                    cameraReader.FrameHeight,
                    cameraReader.FrameCount,
                    cameraReader.PermissionGranted ? "yes" : "no");
            }

            string cameraStatus = cameraReader != null ? cameraReader.LastStatus : string.Empty;
            string detectorStatus = _detector.LastStatus;
            string error = BuildErrorLine();

            statsText.text = string.Format(
                "Init: {0}\n{1}\n{2}\nDetector: {3}\nFPS: {4:0.0} Faces: {5} Pipeline: {6:0.0} ms ONNX: {7:0.0} ms\nDB: {8} people / {9} embeddings {10}{11}",
                _initStatus,
                cameraLine,
                cameraStatus,
                detectorStatus,
                _fps,
                _latestResults.Count,
                _lastPipelineMs,
                _recognizer.LastInferenceMs,
                GetPersonCount(),
                GetEmbeddingCount(),
                string.IsNullOrEmpty(_lastInteractionStatus) ? string.Empty : "\n" + _lastInteractionStatus,
                string.IsNullOrEmpty(error) ? string.Empty : "\n" + error);
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

            statsText.fontSize = Mathf.Min(statsText.fontSize, 22);
            statsText.horizontalOverflow = HorizontalWrapMode.Overflow;
            statsText.verticalOverflow = VerticalWrapMode.Overflow;

            RectTransform rect = statsText.rectTransform;
            if (rect != null)
                rect.sizeDelta = new Vector2(Mathf.Max(rect.sizeDelta.x, 980f), Mathf.Max(rect.sizeDelta.y, 260f));
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
