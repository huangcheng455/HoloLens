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
        public float borderThickness = 3f;

        readonly List<FaceView> _pool = new List<FaceView>();

        public void SetFaces(IReadOnlyList<RecognizedFace> faces, int frameWidth, int frameHeight)
        {
            if (overlayRoot == null)
                overlayRoot = transform as RectTransform;

            EnsurePool(faces.Count);

            for (int i = 0; i < _pool.Count; i++)
                _pool[i].SetActive(i < faces.Count);

            RectTransform target = previewImage != null ? previewImage.rectTransform : overlayRoot;
            Rect displayRect = target.rect;

            for (int i = 0; i < faces.Count; i++)
            {
                RecognizedFace face = faces[i];
                Color color = face.name == "Unknown" ? unknownColor : knownColor;
                Rect rect = MapPixelRect(face.pixelRect, frameWidth, frameHeight, displayRect);
                _pool[i].Set(rect, color, string.Format("{0} {1:0.00}", face.name, face.similarity), borderThickness);
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

        static Rect MapPixelRect(Rect pixelRect, int frameWidth, int frameHeight, Rect displayRect)
        {
            float xMin = displayRect.xMin + pixelRect.xMin / frameWidth * displayRect.width;
            float xMax = displayRect.xMin + pixelRect.xMax / frameWidth * displayRect.width;
            float yMax = displayRect.yMax - pixelRect.yMin / frameHeight * displayRect.height;
            float yMin = displayRect.yMax - pixelRect.yMax / frameHeight * displayRect.height;
            return Rect.MinMaxRect(xMin, yMin, xMax, yMax);
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
