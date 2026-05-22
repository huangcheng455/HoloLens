using HoloFaceRecognition;
using UnityEditor;
using UnityEditor.Events;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

public static class FaceRecognitionUIBuilder
{
    static readonly Color PanelColor = new Color(0f, 0f, 0f, 0.52f);
    static readonly Color FieldColor = new Color(1f, 1f, 1f, 0.94f);
    static readonly Color TextColor = new Color(0.92f, 0.94f, 0.96f, 1f);
    static readonly Color MutedTextColor = new Color(0.74f, 0.78f, 0.82f, 1f);
    static readonly Color RegisterColor = new Color(0.05f, 0.55f, 0.32f, 0.95f);
    static readonly Color ClearColor = new Color(0.56f, 0.16f, 0.16f, 0.95f);

    [MenuItem("Tools/Face Recognition/Beautify UI")]
    public static void BeautifyUI()
    {
        Canvas canvas = Object.FindObjectOfType<Canvas>();
        FaceRecognitionManager manager = Object.FindObjectOfType<FaceRecognitionManager>();
        FaceOverlayRenderer overlay = Object.FindObjectOfType<FaceOverlayRenderer>();

        if (canvas == null)
        {
            Debug.LogWarning("Beautify UI failed: Canvas was not found.");
            return;
        }

        if (manager == null)
            Debug.LogWarning("FaceRecognitionSystem / FaceRecognitionManager was not found. UI will be created, but manager references cannot be bound automatically.");

        Undo.RecordObject(canvas.gameObject, "Beautify Face Recognition UI");
        ConfigureCanvas(canvas);
        ConfigureCameraPreview();
        ConfigureOverlayRoot(overlay);

        RectTransform statsPanel = BuildStatsPanel(canvas, manager);
        RectTransform registerPanel = BuildRegisterPanel(canvas, manager);

        if (overlay != null)
            StyleOverlay(overlay);

        if (manager != null)
        {
            EditorUtility.SetDirty(manager);
            if (manager.statsText == null)
                Debug.LogWarning("StatsText could not be bound automatically. Drag StatsPanel/StatsText to FaceRecognitionManager.statsText.");
            if (manager.registerNameInput == null)
                Debug.LogWarning("RegisterNameInput could not be bound automatically. Drag RegisterPanel/RegisterNameInput to FaceRecognitionManager.registerNameInput.");
            if (manager.statusText == null)
                Debug.LogWarning("RegisterStatusText could not be bound automatically. Drag RegisterPanel/RegisterStatusText to FaceRecognitionManager.statusText.");
            if (manager.clearDatabaseButton == null)
                Debug.LogWarning("ClearDatabaseButton could not be bound automatically. Drag RegisterPanel/ClearDatabaseButton to FaceRecognitionManager.clearDatabaseButton.");
        }

        EditorUtility.SetDirty(canvas.gameObject);
        if (statsPanel != null)
            EditorUtility.SetDirty(statsPanel.gameObject);
        if (registerPanel != null)
            EditorUtility.SetDirty(registerPanel.gameObject);

        EditorSceneManager.MarkSceneDirty(canvas.gameObject.scene);
        Debug.Log("Face Recognition UI beautified. Review the Canvas, then save the scene.");
    }

    static void ConfigureCanvas(Canvas canvas)
    {
        CanvasScaler scaler = canvas.GetComponent<CanvasScaler>();
        if (scaler != null)
        {
            Undo.RecordObject(scaler, "Configure Canvas Scaler");
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1280f, 720f);
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 0.5f;
            EditorUtility.SetDirty(scaler);
        }
    }

    static void ConfigureCameraPreview()
    {
        GameObject previewObject = GameObject.Find("CameraPreview");
        if (previewObject == null)
        {
            Debug.LogWarning("CameraPreview was not found. Camera preview layout was not changed.");
            return;
        }

        RectTransform rect = previewObject.GetComponent<RectTransform>();
        if (rect == null)
            return;

        Undo.RecordObject(rect, "Configure CameraPreview");
        StretchFull(rect);

        AspectRatioFitter fitter = previewObject.GetComponent<AspectRatioFitter>();
        if (fitter != null)
        {
            Undo.RecordObject(fitter, "Configure CameraPreview Aspect");
            fitter.aspectMode = AspectRatioFitter.AspectMode.EnvelopeParent;
            EditorUtility.SetDirty(fitter);
        }

        EditorUtility.SetDirty(previewObject);
    }

    static void ConfigureOverlayRoot(FaceOverlayRenderer overlay)
    {
        GameObject overlayObject = GameObject.Find("OverlayRoot");
        if (overlayObject == null)
        {
            Debug.LogWarning("OverlayRoot was not found. Face overlay layout was not changed.");
            return;
        }

        RectTransform rect = overlayObject.GetComponent<RectTransform>();
        if (rect != null)
        {
            Undo.RecordObject(rect, "Configure OverlayRoot");
            StretchFull(rect);
        }

        if (overlay != null && overlay.overlayRoot == null)
            overlay.overlayRoot = rect;

        EditorUtility.SetDirty(overlayObject);
    }

    static RectTransform BuildStatsPanel(Canvas canvas, FaceRecognitionManager manager)
    {
        RectTransform panel = GetOrCreatePanel(canvas.transform, "StatsPanel");
        Undo.RecordObject(panel, "Configure StatsPanel");
        panel.anchorMin = new Vector2(0f, 1f);
        panel.anchorMax = new Vector2(0f, 1f);
        panel.pivot = new Vector2(0f, 1f);
        panel.anchoredPosition = new Vector2(18f, -18f);
        panel.sizeDelta = new Vector2(260f, 126f);

        Image background = panel.GetComponent<Image>();
        background.color = PanelColor;

        Text statsText = manager != null && manager.statsText != null ? manager.statsText : FindText("StatsText");
        if (statsText == null)
            statsText = CreateText(panel, "StatsText");
        else
            MoveTo(statsText.transform, panel);

        statsText.name = "StatsText";
        StyleText(statsText, 22, TextAnchor.UpperLeft, TextColor);
        statsText.text = "FPS: 0.0\nFaces: 0\nPipeline: 0.0 ms\nONNX: 0.0 ms";
        RectTransform textRect = statsText.rectTransform;
        StretchWithPadding(textRect, 16f, 12f, 16f, 12f);

        if (manager != null)
            manager.statsText = statsText;

        return panel;
    }

    static RectTransform BuildRegisterPanel(Canvas canvas, FaceRecognitionManager manager)
    {
        RectTransform panel = GetOrCreatePanel(canvas.transform, "RegisterPanel");
        Undo.RecordObject(panel, "Configure RegisterPanel");
        panel.anchorMin = new Vector2(0f, 0f);
        panel.anchorMax = new Vector2(0f, 0f);
        panel.pivot = new Vector2(0f, 0f);
        panel.anchoredPosition = new Vector2(18f, 18f);
        panel.sizeDelta = new Vector2(720f, 96f);

        Image background = panel.GetComponent<Image>();
        background.color = PanelColor;

        InputField input = manager != null && manager.registerNameInput != null ? manager.registerNameInput : FindInputField("RegisterNameInput");
        if (input == null)
            input = CreateInputField(panel, "RegisterNameInput");
        else
            MoveTo(input.transform, panel);
        ConfigureInputField(input);
        SetRect(input.GetComponent<RectTransform>(), new Vector2(18f, 24f), new Vector2(250f, 48f));

        Button registerButton = FindButton("RegisterButton");
        if (registerButton == null)
            registerButton = CreateButton(panel, "RegisterButton");
        else
            MoveTo(registerButton.transform, panel);
        ConfigureButton(registerButton, "Register", RegisterColor, Color.white);
        SetRect(registerButton.GetComponent<RectTransform>(), new Vector2(284f, 24f), new Vector2(128f, 48f));

        Button clearButton = manager != null && manager.clearDatabaseButton != null ? manager.clearDatabaseButton : FindButton("ClearDatabaseButton");
        if (clearButton == null)
            clearButton = CreateButton(panel, "ClearDatabaseButton");
        else
            MoveTo(clearButton.transform, panel);
        ConfigureButton(clearButton, "Clear DB", ClearColor, Color.white);
        SetRect(clearButton.GetComponent<RectTransform>(), new Vector2(426f, 24f), new Vector2(128f, 48f));

        Text status = manager != null && manager.statusText != null ? manager.statusText : FindText("RegisterStatusText");
        if (status == null)
            status = CreateText(panel, "RegisterStatusText");
        else
            MoveTo(status.transform, panel);
        status.name = "RegisterStatusText";
        StyleText(status, 22, TextAnchor.MiddleLeft, MutedTextColor);
        status.text = "Ready";
        SetRect(status.rectTransform, new Vector2(574f, 24f), new Vector2(132f, 48f));

        if (manager != null)
        {
            manager.registerNameInput = input;
            manager.statusText = status;
            manager.clearDatabaseButton = clearButton;
            BindButton(registerButton, manager, manager.RegisterCurrentFaceFromUI, "RegisterCurrentFaceFromUI");
            BindButton(clearButton, manager, manager.ClearFaceDatabaseFromUI, "ClearFaceDatabaseFromUI");
        }

        return panel;
    }

    static void StyleOverlay(FaceOverlayRenderer overlay)
    {
        Undo.RecordObject(overlay, "Style Face Overlay");
        overlay.knownColor = new Color(0.1f, 1f, 0.35f, 1f);
        overlay.unknownColor = new Color(1f, 0.68f, 0.08f, 1f);
        overlay.borderThickness = 3f;
        EditorUtility.SetDirty(overlay);
    }

    static RectTransform GetOrCreatePanel(Transform parent, string name)
    {
        GameObject existing = GameObject.Find(name);
        GameObject panelObject = existing != null ? existing : new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        if (existing == null)
            Undo.RegisterCreatedObjectUndo(panelObject, "Create " + name);
        MoveTo(panelObject.transform, parent);
        SetLayerRecursive(panelObject, parent.gameObject.layer);
        Image image = panelObject.GetComponent<Image>();
        if (image == null)
            image = panelObject.AddComponent<Image>();
        image.raycastTarget = false;
        return panelObject.GetComponent<RectTransform>();
    }

    static InputField CreateInputField(Transform parent, string name)
    {
        GameObject root = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(InputField));
        Undo.RegisterCreatedObjectUndo(root, "Create " + name);
        MoveTo(root.transform, parent);
        SetLayerRecursive(root, parent.gameObject.layer);

        Text placeholder = CreateText(root.transform as RectTransform, "Placeholder");
        Text text = CreateText(root.transform as RectTransform, "Text");

        InputField input = root.GetComponent<InputField>();
        input.textComponent = text;
        input.placeholder = placeholder;
        return input;
    }

    static void ConfigureInputField(InputField input)
    {
        input.name = "RegisterNameInput";
        Image image = input.GetComponent<Image>();
        if (image != null)
            image.color = FieldColor;

        Text text = input.textComponent;
        if (text == null)
        {
            text = input.GetComponentInChildren<Text>();
            input.textComponent = text;
        }
        if (text != null)
        {
            StyleText(text, 22, TextAnchor.MiddleLeft, new Color(0.12f, 0.14f, 0.16f, 1f));
            StretchWithPadding(text.rectTransform, 12f, 4f, 12f, 4f);
        }

        Text placeholder = input.placeholder as Text;
        if (placeholder == null)
        {
            Transform placeholderTransform = input.transform.Find("Placeholder");
            placeholder = placeholderTransform == null ? CreateText(input.transform, "Placeholder") : placeholderTransform.GetComponent<Text>();
            input.placeholder = placeholder;
        }
        if (placeholder != null)
        {
            StyleText(placeholder, 22, TextAnchor.MiddleLeft, new Color(0.34f, 0.38f, 0.42f, 0.82f));
            placeholder.fontStyle = FontStyle.Italic;
            placeholder.text = "Enter name";
            StretchWithPadding(placeholder.rectTransform, 12f, 4f, 12f, 4f);
        }
    }

    static Button CreateButton(Transform parent, string name)
    {
        GameObject root = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Button));
        Undo.RegisterCreatedObjectUndo(root, "Create " + name);
        MoveTo(root.transform, parent);
        SetLayerRecursive(root, parent.gameObject.layer);
        CreateText(root.transform, "Text");
        return root.GetComponent<Button>();
    }

    static void ConfigureButton(Button button, string label, Color backgroundColor, Color labelColor)
    {
        button.name = label == "Clear DB" ? "ClearDatabaseButton" : "RegisterButton";
        Image image = button.GetComponent<Image>();
        if (image == null)
            image = button.gameObject.AddComponent<Image>();
        image.color = backgroundColor;
        image.raycastTarget = true;
        button.targetGraphic = image;

        ColorBlock colors = button.colors;
        colors.normalColor = backgroundColor;
        colors.highlightedColor = Lighten(backgroundColor, 0.12f);
        colors.pressedColor = Darken(backgroundColor, 0.12f);
        colors.selectedColor = Lighten(backgroundColor, 0.08f);
        colors.disabledColor = new Color(0.38f, 0.38f, 0.38f, 0.55f);
        colors.colorMultiplier = 1f;
        colors.fadeDuration = 0.08f;
        button.colors = colors;

        Text text = button.GetComponentInChildren<Text>();
        if (text == null)
            text = CreateText(button.transform, "Text");
        StyleText(text, 22, TextAnchor.MiddleCenter, labelColor);
        text.text = label;
        StretchWithPadding(text.rectTransform, 4f, 4f, 4f, 4f);
    }

    static Text CreateText(Transform parent, string name)
    {
        GameObject textObject = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        Undo.RegisterCreatedObjectUndo(textObject, "Create " + name);
        MoveTo(textObject.transform, parent);
        SetLayerRecursive(textObject, parent.gameObject.layer);
        Text text = textObject.GetComponent<Text>();
        text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        return text;
    }

    static void StyleText(Text text, int fontSize, TextAnchor alignment, Color color)
    {
        Undo.RecordObject(text, "Style Text");
        text.font = text.font != null ? text.font : Resources.GetBuiltinResource<Font>("Arial.ttf");
        text.fontSize = fontSize;
        text.alignment = alignment;
        text.color = color;
        text.horizontalOverflow = HorizontalWrapMode.Overflow;
        text.verticalOverflow = VerticalWrapMode.Overflow;
        text.raycastTarget = false;
        EditorUtility.SetDirty(text);
    }

    static void BindButton(Button button, FaceRecognitionManager manager, UnityAction action, string methodName)
    {
        if (button == null || manager == null)
            return;

        Undo.RecordObject(button, "Bind " + button.name);
        RemovePersistentCalls(button, manager, methodName == "RegisterCurrentFaceFromUI" ? "RegisterCurrentFace" : methodName);
        RemovePersistentCalls(button, manager, methodName);
        UnityEventTools.AddPersistentListener(button.onClick, action);
        EditorUtility.SetDirty(button);
    }

    static void RemovePersistentCalls(Button button, Object target, string methodName)
    {
        UnityEvent buttonEvent = button.onClick;
        for (int i = buttonEvent.GetPersistentEventCount() - 1; i >= 0; i--)
        {
            if (buttonEvent.GetPersistentTarget(i) == target && buttonEvent.GetPersistentMethodName(i) == methodName)
                UnityEventTools.RemovePersistentListener(buttonEvent, i);
        }
    }

    static Text FindText(string name)
    {
        GameObject obj = GameObject.Find(name);
        return obj == null ? null : obj.GetComponent<Text>();
    }

    static InputField FindInputField(string name)
    {
        GameObject obj = GameObject.Find(name);
        return obj == null ? null : obj.GetComponent<InputField>();
    }

    static Button FindButton(string name)
    {
        GameObject obj = GameObject.Find(name);
        return obj == null ? null : obj.GetComponent<Button>();
    }

    static void MoveTo(Transform child, Transform parent)
    {
        if (child.parent == parent)
            return;

        Undo.SetTransformParent(child, parent, "Move " + child.name);
        child.SetAsLastSibling();
    }

    static void SetRect(RectTransform rect, Vector2 position, Vector2 size)
    {
        Undo.RecordObject(rect, "Layout " + rect.name);
        rect.anchorMin = new Vector2(0f, 0f);
        rect.anchorMax = new Vector2(0f, 0f);
        rect.pivot = new Vector2(0f, 0f);
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
        EditorUtility.SetDirty(rect);
    }

    static void StretchFull(RectTransform rect)
    {
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.offsetMin = Vector2.zero;
        rect.offsetMax = Vector2.zero;
    }

    static void StretchWithPadding(RectTransform rect, float left, float bottom, float right, float top)
    {
        Undo.RecordObject(rect, "Stretch " + rect.name);
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.offsetMin = new Vector2(left, bottom);
        rect.offsetMax = new Vector2(-right, -top);
        EditorUtility.SetDirty(rect);
    }

    static void SetLayerRecursive(GameObject root, int layer)
    {
        root.layer = layer;
        foreach (Transform child in root.transform)
            SetLayerRecursive(child.gameObject, layer);
    }

    static Color Lighten(Color color, float amount)
    {
        return new Color(
            Mathf.Clamp01(color.r + amount),
            Mathf.Clamp01(color.g + amount),
            Mathf.Clamp01(color.b + amount),
            color.a);
    }

    static Color Darken(Color color, float amount)
    {
        return new Color(
            Mathf.Clamp01(color.r - amount),
            Mathf.Clamp01(color.g - amount),
            Mathf.Clamp01(color.b - amount),
            color.a);
    }
}
