using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace HoloFaceRecognition
{
    public sealed class FaceOverlayRenderer : MonoBehaviour
    {
        public RectTransform overlayRoot;
        public RawImage previewImage;
        public Font labelFont;
        public Color knownColor = new Color(0.1f, 1f, 0.35f, 1f);
        public Color unknownColor = new Color(1f, 0.68f, 0.08f, 1f);
        public Color debugColor = new Color(0.15f, 0.75f, 1f, 1f);
        public float borderThickness = 3f;
        [Range(0f, 0.5f)] public float overlayBoxPadding = 0.18f;
        public bool drawDebugCenterTestBox;
        public bool applyCameraOrientationToOverlay;
        public bool pixelRectOriginBottomLeft = true;

        readonly List<FaceView> _pool = new List<FaceView>();

        public void SetFaces(IReadOnlyList<RecognizedFace> faces, CameraFrame frame)
        {
            if (frame == null)
            {
                SetFaces(faces, 0, 0);
                return;
            }

            int overlayRotation = applyCameraOrientationToOverlay ? frame.rotationAngle : 0;
            bool overlayMirrored = applyCameraOrientationToOverlay && frame.verticallyMirrored;
            SetFaces(faces, frame.width, frame.height, overlayRotation, overlayMirrored);
        }

        public void SetFaces(IReadOnlyList<RecognizedFace> faces, int frameWidth, int frameHeight)
        {
            SetFaces(faces, frameWidth, frameHeight, 0, false);
        }

        public void SetFaces(IReadOnlyList<RecognizedFace> faces, int frameWidth, int frameHeight, int rotationAngle, bool verticallyMirrored)
        {
            if (overlayRoot == null)
                overlayRoot = transform as RectTransform;

            int faceCount = faces != null ? faces.Count : 0;
            int viewCount = faceCount + (drawDebugCenterTestBox ? 1 : 0);
            EnsurePool(viewCount);

            for (int i = 0; i < _pool.Count; i++)
                _pool[i].SetActive(i < viewCount);

            RectTransform target = previewImage != null ? previewImage.rectTransform : overlayRoot;
            Rect displayRect = GetRectInOverlaySpace(target);
            Rect contentRect = GetDisplayedTextureRect(displayRect, target, frameWidth, frameHeight, rotationAngle);

            for (int i = 0; i < faceCount; i++)
            {
                RecognizedFace face = faces[i];
                bool hasRecognitionScore = face.similarity > 0f;
                Color color = face.name == "Unknown" || !hasRecognitionScore ? unknownColor : knownColor;
                Rect paddedPixelRect = AddPixelPadding(face.pixelRect, overlayBoxPadding, frameWidth, frameHeight);
                Rect rect = MapPixelRect(paddedPixelRect, frameWidth, frameHeight, contentRect, rotationAngle, verticallyMirrored, pixelRectOriginBottomLeft);
                string label = hasRecognitionScore ? string.Format("{0} {1:0.00}", face.name, face.similarity) : face.name;
                _pool[i].Set(rect, color, label, borderThickness);
            }

            if (drawDebugCenterTestBox)
            {
                float width = contentRect.width * 0.5f;
                float height = contentRect.height * 0.5f;
                Rect debugRect = Rect.MinMaxRect(
                    contentRect.center.x - width * 0.5f,
                    contentRect.center.y - height * 0.5f,
                    contentRect.center.x + width * 0.5f,
                    contentRect.center.y + height * 0.5f);
                _pool[faceCount].Set(debugRect, debugColor, "DEBUG 50%", borderThickness);
            }
        }

        void EnsurePool(int count)
        {
            while (_pool.Count < count)
                _pool.Add(CreateFaceView("FaceOverlay_" + _pool.Count));
        }

        FaceView CreateFaceView(string objectName)
        {
            var root = new GameObject(objectName, typeof(RectTransform));
            root.transform.SetParent(overlayRoot != null ? overlayRoot : transform, false);
            return new FaceView(root.transform as RectTransform, labelFont);
        }

        Rect GetRectInOverlaySpace(RectTransform target)
        {
            if (target == null)
                return Rect.zero;

            if (overlayRoot == null || target == overlayRoot)
                return target.rect;

            Vector3[] corners = new Vector3[4];
            target.GetWorldCorners(corners);

            Vector2 min = overlayRoot.InverseTransformPoint(corners[0]);
            Vector2 max = min;
            for (int i = 1; i < corners.Length; i++)
            {
                Vector2 local = overlayRoot.InverseTransformPoint(corners[i]);
                min = Vector2.Min(min, local);
                max = Vector2.Max(max, local);
            }

            return Rect.MinMaxRect(min.x, min.y, max.x, max.y);
        }

        static Rect GetDisplayedTextureRect(Rect displayRect, RectTransform target, int frameWidth, int frameHeight, int rotationAngle)
        {
            AspectRatioFitter fitter = target == null ? null : target.GetComponent<AspectRatioFitter>();
            if (fitter != null && fitter.enabled)
            {
                switch (fitter.aspectMode)
                {
                    case AspectRatioFitter.AspectMode.WidthControlsHeight:
                    case AspectRatioFitter.AspectMode.HeightControlsWidth:
                    case AspectRatioFitter.AspectMode.FitInParent:
                    case AspectRatioFitter.AspectMode.EnvelopeParent:
                        return displayRect;
                }
            }

            return GetAspectFitRect(displayRect, frameWidth, frameHeight, rotationAngle);
        }

        static Rect GetAspectFitRect(Rect displayRect, int frameWidth, int frameHeight, int rotationAngle)
        {
            if (frameWidth <= 0 || frameHeight <= 0 || displayRect.width <= 0f || displayRect.height <= 0f)
                return displayRect;

            int normalizedRotation = NormalizeRotation(rotationAngle);
            float orientedWidth = normalizedRotation == 90 || normalizedRotation == 270 ? frameHeight : frameWidth;
            float orientedHeight = normalizedRotation == 90 || normalizedRotation == 270 ? frameWidth : frameHeight;
            float imageAspect = orientedWidth / orientedHeight;
            float displayAspect = displayRect.width / displayRect.height;

            if (displayAspect > imageAspect)
            {
                float width = displayRect.height * imageAspect;
                float xMin = displayRect.center.x - width * 0.5f;
                return new Rect(xMin, displayRect.yMin, width, displayRect.height);
            }

            float height = displayRect.width / imageAspect;
            float yMin = displayRect.center.y - height * 0.5f;
            return new Rect(displayRect.xMin, yMin, displayRect.width, height);
        }

        static Rect MapPixelRect(Rect pixelRect, int frameWidth, int frameHeight, Rect displayRect, int rotationAngle, bool verticallyMirrored, bool originBottomLeft)
        {
            if (frameWidth <= 0 || frameHeight <= 0)
                return Rect.zero;

            Vector2[] corners =
            {
                TransformPixelPoint(pixelRect.xMin, pixelRect.yMin, frameWidth, frameHeight, rotationAngle, verticallyMirrored),
                TransformPixelPoint(pixelRect.xMax, pixelRect.yMin, frameWidth, frameHeight, rotationAngle, verticallyMirrored),
                TransformPixelPoint(pixelRect.xMax, pixelRect.yMax, frameWidth, frameHeight, rotationAngle, verticallyMirrored),
                TransformPixelPoint(pixelRect.xMin, pixelRect.yMax, frameWidth, frameHeight, rotationAngle, verticallyMirrored)
            };

            Vector2 min = corners[0];
            Vector2 max = corners[0];
            for (int i = 1; i < corners.Length; i++)
            {
                min = Vector2.Min(min, corners[i]);
                max = Vector2.Max(max, corners[i]);
            }

            float xMin = displayRect.xMin + Mathf.Clamp01(min.x) * displayRect.width;
            float xMax = displayRect.xMin + Mathf.Clamp01(max.x) * displayRect.width;
            float yMin;
            float yMax;
            if (originBottomLeft)
            {
                yMin = displayRect.yMin + Mathf.Clamp01(min.y) * displayRect.height;
                yMax = displayRect.yMin + Mathf.Clamp01(max.y) * displayRect.height;
            }
            else
            {
                yMax = displayRect.yMax - Mathf.Clamp01(min.y) * displayRect.height;
                yMin = displayRect.yMax - Mathf.Clamp01(max.y) * displayRect.height;
            }
            return Rect.MinMaxRect(xMin, yMin, xMax, yMax);
        }

        static Rect AddPixelPadding(Rect rect, float padding, int frameWidth, int frameHeight)
        {
            if (padding <= 0f)
                return rect;

            float padX = rect.width * padding;
            float padY = rect.height * padding;
            float xMin = Mathf.Clamp(rect.xMin - padX, 0f, frameWidth);
            float xMax = Mathf.Clamp(rect.xMax + padX, 0f, frameWidth);
            float yMin = Mathf.Clamp(rect.yMin - padY, 0f, frameHeight);
            float yMax = Mathf.Clamp(rect.yMax + padY, 0f, frameHeight);
            return Rect.MinMaxRect(xMin, yMin, xMax, yMax);
        }

        static Vector2 TransformPixelPoint(float pixelX, float pixelY, int frameWidth, int frameHeight, int rotationAngle, bool verticallyMirrored)
        {
            float x = pixelX / frameWidth;
            float y = pixelY / frameHeight;

            if (verticallyMirrored)
                y = 1f - y;

            switch (NormalizeRotation(rotationAngle))
            {
                case 90:
                    return new Vector2(1f - y, x);
                case 180:
                    return new Vector2(1f - x, 1f - y);
                case 270:
                    return new Vector2(y, 1f - x);
                default:
                    return new Vector2(x, y);
            }
        }

        static int NormalizeRotation(int rotationAngle)
        {
            int normalized = rotationAngle % 360;
            if (normalized < 0)
                normalized += 360;

            if (normalized < 45 || normalized >= 315)
                return 0;
            if (normalized < 135)
                return 90;
            if (normalized < 225)
                return 180;
            return 270;
        }

        sealed class FaceView
        {
            readonly RectTransform _root;
            readonly Image[] _lines = new Image[4];
            readonly Image _labelBackground;
            readonly Text _label;

            public FaceView(RectTransform root, Font font)
            {
                _root = root;
                _root.anchorMin = new Vector2(0.5f, 0.5f);
                _root.anchorMax = new Vector2(0.5f, 0.5f);

                for (int i = 0; i < _lines.Length; i++)
                {
                    var line = new GameObject("Line_" + i, typeof(RectTransform), typeof(Image));
                    line.transform.SetParent(_root, false);
                    _lines[i] = line.GetComponent<Image>();
                }

                var labelBackgroundObject = new GameObject("LabelBackground", typeof(RectTransform), typeof(Image));
                labelBackgroundObject.transform.SetParent(_root, false);
                _labelBackground = labelBackgroundObject.GetComponent<Image>();
                _labelBackground.color = new Color(0f, 0f, 0f, 0.62f);

                var labelObject = new GameObject("Label", typeof(RectTransform), typeof(Text));
                labelObject.transform.SetParent(_root, false);
                _label = labelObject.GetComponent<Text>();
                _label.font = font != null ? font : Resources.GetBuiltinResource<Font>("Arial.ttf");
                _label.fontSize = 22;
                _label.alignment = TextAnchor.MiddleLeft;
                _label.horizontalOverflow = HorizontalWrapMode.Overflow;
                _label.verticalOverflow = VerticalWrapMode.Overflow;
            }

            public void SetActive(bool active)
            {
                _root.gameObject.SetActive(active);
            }

            public void Set(Rect rect, Color color, string label, float thickness)
            {
                _root.anchoredPosition = rect.center;
                _root.sizeDelta = rect.size;

                foreach (var line in _lines)
                    line.color = color;

                SetLine(_lines[0].rectTransform, new Vector2(0.5f, 1f), new Vector2(rect.width, thickness), Vector2.zero);
                SetLine(_lines[1].rectTransform, new Vector2(0.5f, 0f), new Vector2(rect.width, thickness), Vector2.zero);
                SetLine(_lines[2].rectTransform, new Vector2(0f, 0.5f), new Vector2(thickness, rect.height), Vector2.zero);
                SetLine(_lines[3].rectTransform, new Vector2(1f, 0.5f), new Vector2(thickness, rect.height), Vector2.zero);

                _label.text = label;
                _label.color = color;

                var backgroundRect = _labelBackground.rectTransform;
                backgroundRect.anchorMin = new Vector2(0f, 1f);
                backgroundRect.anchorMax = new Vector2(0f, 1f);
                backgroundRect.pivot = new Vector2(0f, 0f);
                backgroundRect.anchoredPosition = new Vector2(0f, 4f);
                backgroundRect.sizeDelta = new Vector2(220f, 32f);

                var labelRect = _label.rectTransform;
                labelRect.anchorMin = new Vector2(0f, 1f);
                labelRect.anchorMax = new Vector2(0f, 1f);
                labelRect.pivot = new Vector2(0f, 0f);
                labelRect.anchoredPosition = new Vector2(8f, 4f);
                labelRect.sizeDelta = new Vector2(204f, 32f);
            }

            static void SetLine(RectTransform rect, Vector2 anchor, Vector2 size, Vector2 offset)
            {
                rect.anchorMin = anchor;
                rect.anchorMax = anchor;
                rect.pivot = anchor;
                rect.anchoredPosition = offset;
                rect.sizeDelta = size;
            }
        }
    }
}
