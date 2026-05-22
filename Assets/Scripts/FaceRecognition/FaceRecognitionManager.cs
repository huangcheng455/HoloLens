using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

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

        readonly FaceAligner _aligner = new FaceAligner();
        readonly FaceMatcher _matcher = new FaceMatcher();
        readonly FaceDetectorService _detector = new FaceDetectorService();
        readonly OnnxFaceRecognizer _recognizer = new OnnxFaceRecognizer();
        readonly FaceDatabase _database = new FaceDatabase();
        readonly List<RecognizedFace> _latestResults = new List<RecognizedFace>();

        bool _initialized;
        bool _processing;
        int _frameCounter;
        float _fpsTimer;
        int _fpsFrames;
        float _fps;
        float _lastPipelineMs;
        float[] _lastEmbedding;
        GameObject _popupRoot;
        Text _popupText;
        Coroutine _popupRoutine;

        void Awake()
        {
            EnsureRegistrationUi();
        }

        async void Start()
        {
            await InitializeAsync();
        }

        void OnDestroy()
        {
            _recognizer.Dispose();
        }

        void Update()
        {
            _matcher.Threshold = threshold;
            UpdateFps();

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
            await _database.LoadAsync();
            await _detector.InitializeAsync();

            string modelPath = await PrepareModelFileAsync(modelFileName);
            await _recognizer.InitializeAsync(modelPath, modelName);
            _initialized = true;
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
            }
            catch (Exception ex)
            {
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
                return;
            }

            if (_lastEmbedding == null)
            {
                UnityEngine.Debug.LogWarning("No face embedding is available yet.");
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
            SetStatusText(string.Empty);
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

            statsText.text = string.Format(
                "FPS: {0:0.0}\nFaces: {1}\nPipeline: {2:0.0} ms\nONNX: {3:0.0} ms",
                _fps,
                _latestResults.Count,
                _lastPipelineMs,
                _recognizer.LastInferenceMs);
        }
    }
}
