using System;
using UnityEngine;

namespace HoloFaceRecognition
{
    public sealed class FaceAligner
    {
        public const int OutputSize = 112;
        readonly Color32[] _outputBuffer = new Color32[OutputSize * OutputSize];

        static readonly Vector2[] ArcFace112Template =
        {
            new Vector2(38.2946f, 51.6963f),
            new Vector2(73.5318f, 51.5014f),
            new Vector2(56.0252f, 71.7366f),
            new Vector2(41.5493f, 92.3655f),
            new Vector2(70.7299f, 92.2041f)
        };

        public Color32[] Align(CameraFrame frame, FaceDetectionResult face, float bboxPadding = 0.25f)
        {
            if (face.landmarks != null && face.landmarks.Length >= 5)
                return AlignByLandmarks(frame.pixels, frame.width, frame.height, face.landmarks);

            return AlignByBoundingBox(frame.pixels, frame.width, frame.height, face.pixelRect, bboxPadding);
        }

        public Color32[] AlignByBoundingBox(Color32[] source, int width, int height, Rect bbox, float padding)
        {
            float side = Mathf.Max(bbox.width, bbox.height) * (1f + padding * 2f);
            float cx = bbox.x + bbox.width * 0.5f;
            float cy = bbox.y + bbox.height * 0.5f;
            var square = new Rect(cx - side * 0.5f, cy - side * 0.5f, side, side);

            for (int y = 0; y < OutputSize; y++)
            {
                float sy = square.y + (y + 0.5f) * square.height / OutputSize;
                for (int x = 0; x < OutputSize; x++)
                {
                    float sx = square.x + (x + 0.5f) * square.width / OutputSize;
                    _outputBuffer[y * OutputSize + x] = SampleBilinear(source, width, height, sx, sy);
                }
            }
            return _outputBuffer;
        }

        public Color32[] AlignByLandmarks(Color32[] source, int width, int height, Vector2[] landmarks)
        {
            SimilarityTransform transform = EstimateSimilarity(ArcFace112Template, landmarks);

            for (int y = 0; y < OutputSize; y++)
            {
                for (int x = 0; x < OutputSize; x++)
                {
                    Vector2 src = transform.TransformPoint(new Vector2(x, y));
                    _outputBuffer[y * OutputSize + x] = SampleBilinear(source, width, height, src.x, src.y);
                }
            }

            return _outputBuffer;
        }

        static SimilarityTransform EstimateSimilarity(Vector2[] dst, Vector2[] src)
        {
            int count = Math.Min(5, Math.Min(dst.Length, src.Length));
            Vector2 srcMean = Vector2.zero;
            Vector2 dstMean = Vector2.zero;
            for (int i = 0; i < count; i++)
            {
                srcMean += src[i];
                dstMean += dst[i];
            }
            srcMean /= count;
            dstMean /= count;

            float a = 0f;
            float b = 0f;
            float denom = 0f;
            for (int i = 0; i < count; i++)
            {
                Vector2 s = src[i] - srcMean;
                Vector2 d = dst[i] - dstMean;
                a += s.x * d.x + s.y * d.y;
                b += s.y * d.x - s.x * d.y;
                denom += d.sqrMagnitude;
            }

            if (denom < 1e-6f)
                return SimilarityTransform.Identity;

            a /= denom;
            b /= denom;
            Vector2 t = srcMean - new Vector2(a * dstMean.x + b * dstMean.y, -b * dstMean.x + a * dstMean.y);
            return new SimilarityTransform(a, b, t);
        }

        static Color32 SampleBilinear(Color32[] pixels, int width, int height, float x, float y)
        {
            x = Mathf.Clamp(x, 0, width - 1);
            y = Mathf.Clamp(y, 0, height - 1);

            int x0 = Mathf.FloorToInt(x);
            int y0 = Mathf.FloorToInt(y);
            int x1 = Mathf.Min(x0 + 1, width - 1);
            int y1 = Mathf.Min(y0 + 1, height - 1);
            float tx = x - x0;
            float ty = y - y0;

            Color c00 = pixels[y0 * width + x0];
            Color c10 = pixels[y0 * width + x1];
            Color c01 = pixels[y1 * width + x0];
            Color c11 = pixels[y1 * width + x1];
            Color c0 = Color.Lerp(c00, c10, tx);
            Color c1 = Color.Lerp(c01, c11, tx);
            return Color.Lerp(c0, c1, ty);
        }

        struct SimilarityTransform
        {
            public static readonly SimilarityTransform Identity = new SimilarityTransform(1f, 0f, Vector2.zero);

            readonly float _a;
            readonly float _b;
            readonly Vector2 _t;

            public SimilarityTransform(float a, float b, Vector2 t)
            {
                _a = a;
                _b = b;
                _t = t;
            }

            public Vector2 TransformPoint(Vector2 p)
            {
                return new Vector2(_a * p.x + _b * p.y + _t.x, -_b * p.x + _a * p.y + _t.y);
            }
        }
    }
}
