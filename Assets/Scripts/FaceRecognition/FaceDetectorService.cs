using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

#if WINDOWS_UWP && !UNITY_EDITOR
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Graphics.Imaging;
using Windows.Media.FaceAnalysis;
#endif

namespace HoloFaceRecognition
{
    [Serializable]
    public sealed class FaceDetectionResult
    {
        public Rect pixelRect;
        public Vector2[] landmarks;
        public float confidence = 1f;
    }

    public interface IFaceDetectorService
    {
        Task InitializeAsync();
        Task<List<FaceDetectionResult>> DetectAsync(CameraFrame frame);
    }

    public sealed class FaceDetectorService : IFaceDetectorService
    {
        public string LastStatus { get; private set; } = "Detector not initialized";
        public string LastError { get; private set; } = string.Empty;
        public int LastDetectedFaceCount { get; private set; }
        string _lastLoggedStatus = string.Empty;

#if WINDOWS_UWP && !UNITY_EDITOR
        const int DetectionLongSide = 320;
        const int RotationFallbackAfterMisses = 3;

        FaceDetector _detector;
        int _lastSuccessfulRotation = int.MinValue;
        int _missesSinceRotationHit;
#endif

        public async Task InitializeAsync()
        {
#if WINDOWS_UWP && !UNITY_EDITOR
            try
            {
                _detector = await FaceDetector.CreateAsync();
                LastStatus = "UWP FaceDetector initialized";
                LastError = string.Empty;
                LogStatusChanged();
            }
            catch (Exception ex)
            {
                LastStatus = "UWP FaceDetector init failed";
                LastError = ex.GetType().Name + ": " + ex.Message;
                UnityEngine.Debug.LogException(ex);
                throw;
            }
#else
            LastStatus = "Mock detector active";
            LastError = string.Empty;
            LogStatusChanged();
            await Task.CompletedTask;
#endif
        }

        public async Task<List<FaceDetectionResult>> DetectAsync(CameraFrame frame)
        {
            var results = new List<FaceDetectionResult>();
            LastDetectedFaceCount = 0;

            if (frame == null || frame.pixels == null || frame.width <= 0 || frame.height <= 0)
            {
                LastStatus = "Detector skipped: invalid frame";
                return results;
            }

#if WINDOWS_UWP && !UNITY_EDITOR
            try
            {
                if (_detector == null)
                    await InitializeAsync();

                int rotationUsed = 0;
                DetectionBitmapInfo detectionBitmapInfo = new DetectionBitmapInfo();
                IList<DetectedFace> detectedFaces = null;
                int[] rotationAttempts = BuildRotationAttempts(frame.rotationAngle);
                var attemptSummary = new List<string>();
                foreach (int rotation in rotationAttempts)
                {
                    using (SoftwareBitmap bitmap = CreateGray8SoftwareBitmap(frame, rotation, out detectionBitmapInfo))
                    {
                        detectedFaces = await _detector.DetectFacesAsync(bitmap);
                        int faceCount = detectedFaces == null ? 0 : detectedFaces.Count;
                        attemptSummary.Add(rotation + ":" + faceCount);
                        rotationUsed = rotation;
                        if (faceCount > 0)
                            break;
                    }
                }

                if (detectedFaces != null && detectedFaces.Count > 0)
                {
                    _lastSuccessfulRotation = rotationUsed;
                    _missesSinceRotationHit = 0;
                }
                else
                {
                    _missesSinceRotationHit++;
                }

                if (detectedFaces != null)
                {
                    foreach (var face in detectedFaces)
                    {
                        BitmapBounds box = face.FaceBox;
                        Rect pixelRect = MapDetectedRectToFrame(
                            new Rect(box.X, box.Y, box.Width, box.Height),
                            detectionBitmapInfo.width,
                            detectionBitmapInfo.height,
                            frame.width,
                            frame.height,
                            rotationUsed);
                        results.Add(new FaceDetectionResult
                        {
                            pixelRect = pixelRect,
                            landmarks = null,
                            confidence = 1f
                        });
                    }
                }

                LastDetectedFaceCount = results.Count;
                LastStatus = LastDetectedFaceCount > 0
                    ? "faces detected: " + LastDetectedFaceCount + " rotation=" + rotationUsed + " detect=" + detectionBitmapInfo.width + "x" + detectionBitmapInfo.height + " attempts=" + string.Join(",", attemptSummary.ToArray())
                    : "faces detected: 0 detect=" + detectionBitmapInfo.width + "x" + detectionBitmapInfo.height + " attempts=" + string.Join(",", attemptSummary.ToArray());
                LastError = string.Empty;
                LogStatusChanged();
            }
            catch (Exception ex)
            {
                LastStatus = "UWP detector failed";
                LastError = ex.GetType().Name + ": " + ex.Message;
                UnityEngine.Debug.LogException(ex);
            }
#else
            await Task.CompletedTask;

            // Unity Editor / PC fallback:
            // Return a mock face box in the center of the frame.
            float boxW = frame.width * 0.42f;
            float boxH = frame.height * 0.62f;
            float boxX = (frame.width - boxW) * 0.5f;
            float boxY = (frame.height - boxH) * 0.58f;

            results.Add(new FaceDetectionResult
            {
                pixelRect = new Rect(boxX, boxY, boxW, boxH),
                landmarks = null,
                confidence = 1f
            });

            LastDetectedFaceCount = results.Count;
            LastStatus = "Mock detector active; faces detected: " + LastDetectedFaceCount;
            LastError = string.Empty;
            LogStatusChanged();
#endif

            return results;
        }

        void LogStatusChanged()
        {
            if (LastStatus == _lastLoggedStatus)
                return;

            _lastLoggedStatus = LastStatus;
            UnityEngine.Debug.Log(LastStatus);
        }

#if WINDOWS_UWP && !UNITY_EDITOR
        int[] BuildRotationAttempts(int preferredRotation)
        {
            int normalized = NormalizeRotation(preferredRotation);

            if (_lastSuccessfulRotation != int.MinValue && _missesSinceRotationHit < RotationFallbackAfterMisses)
                return new[] { _lastSuccessfulRotation };

            var rotations = new List<int>();
            AddUniqueRotation(rotations, _lastSuccessfulRotation);
            AddUniqueRotation(rotations, normalized);
            AddUniqueRotation(rotations, 0);
            AddUniqueRotation(rotations, 90);
            AddUniqueRotation(rotations, 270);
            AddUniqueRotation(rotations, 180);
            return rotations.ToArray();
        }

        static void AddUniqueRotation(List<int> rotations, int rotation)
        {
            if (rotation == int.MinValue)
                return;

            int normalized = NormalizeRotation(rotation);
            if (!rotations.Contains(normalized))
                rotations.Add(normalized);
        }

        struct DetectionBitmapInfo
        {
            public int width;
            public int height;
        }

        static SoftwareBitmap CreateGray8SoftwareBitmap(CameraFrame frame, int rotation, out DetectionBitmapInfo bitmapInfo)
        {
            int normalizedRotation = NormalizeRotation(rotation);
            int fullRotatedWidth = normalizedRotation == 90 || normalizedRotation == 270 ? frame.height : frame.width;
            int fullRotatedHeight = normalizedRotation == 90 || normalizedRotation == 270 ? frame.width : frame.height;
            float scale = Mathf.Min(1f, DetectionLongSide / (float)Mathf.Max(fullRotatedWidth, fullRotatedHeight));
            int bitmapWidth = Mathf.Max(1, Mathf.RoundToInt(fullRotatedWidth * scale));
            int bitmapHeight = Mathf.Max(1, Mathf.RoundToInt(fullRotatedHeight * scale));
            bitmapInfo = new DetectionBitmapInfo
            {
                width = bitmapWidth,
                height = bitmapHeight
            };

            var bitmap = new SoftwareBitmap(
                BitmapPixelFormat.Gray8,
                bitmapWidth,
                bitmapHeight,
                BitmapAlphaMode.Ignore
            );

            byte[] bytes = new byte[bitmapWidth * bitmapHeight];

            for (int targetY = 0; targetY < bitmapHeight; targetY++)
            {
                float rotatedY = (targetY + 0.5f) * fullRotatedHeight / bitmapHeight - 0.5f;
                for (int targetX = 0; targetX < bitmapWidth; targetX++)
                {
                    float rotatedX = (targetX + 0.5f) * fullRotatedWidth / bitmapWidth - 0.5f;
                    Vector2 source = MapRotatedPointToFrame(rotatedX, rotatedY, frame.width, frame.height, normalizedRotation);
                    int sourceX = Mathf.Clamp(Mathf.RoundToInt(source.x), 0, frame.width - 1);
                    int sourceY = Mathf.Clamp(Mathf.RoundToInt(source.y), 0, frame.height - 1);
                    Color32 c = frame.pixels[sourceY * frame.width + sourceX];
                    bytes[targetY * bitmapWidth + targetX] = (byte)((c.r * 299 + c.g * 587 + c.b * 114) / 1000);
                }
            }

            bitmap.CopyFromBuffer(bytes.AsBuffer());
            return bitmap;
        }

        static Rect MapDetectedRectToFrame(Rect detectedRect, int detectionWidth, int detectionHeight, int frameWidth, int frameHeight, int rotation)
        {
            int normalizedRotation = NormalizeRotation(rotation);
            int fullRotatedWidth = normalizedRotation == 90 || normalizedRotation == 270 ? frameHeight : frameWidth;
            int fullRotatedHeight = normalizedRotation == 90 || normalizedRotation == 270 ? frameWidth : frameHeight;
            float scaleX = fullRotatedWidth / (float)Mathf.Max(1, detectionWidth);
            float scaleY = fullRotatedHeight / (float)Mathf.Max(1, detectionHeight);
            Rect rotatedRect = new Rect(
                detectedRect.x * scaleX,
                detectedRect.y * scaleY,
                detectedRect.width * scaleX,
                detectedRect.height * scaleY);

            Vector2[] corners =
            {
                MapRotatedPointToFrame(rotatedRect.xMin, rotatedRect.yMin, frameWidth, frameHeight, normalizedRotation),
                MapRotatedPointToFrame(rotatedRect.xMax, rotatedRect.yMin, frameWidth, frameHeight, normalizedRotation),
                MapRotatedPointToFrame(rotatedRect.xMax, rotatedRect.yMax, frameWidth, frameHeight, normalizedRotation),
                MapRotatedPointToFrame(rotatedRect.xMin, rotatedRect.yMax, frameWidth, frameHeight, normalizedRotation)
            };

            Vector2 min = corners[0];
            Vector2 max = corners[0];
            for (int i = 1; i < corners.Length; i++)
            {
                min = Vector2.Min(min, corners[i]);
                max = Vector2.Max(max, corners[i]);
            }

            min.x = Mathf.Clamp(min.x, 0f, frameWidth);
            min.y = Mathf.Clamp(min.y, 0f, frameHeight);
            max.x = Mathf.Clamp(max.x, 0f, frameWidth);
            max.y = Mathf.Clamp(max.y, 0f, frameHeight);
            return Rect.MinMaxRect(min.x, min.y, max.x, max.y);
        }

        static Vector2 MapRotatedPointToFrame(float x, float y, int width, int height, int rotation)
        {
            switch (rotation)
            {
                case 90:
                    return new Vector2(y, height - x);
                case 180:
                    return new Vector2(width - x, height - y);
                case 270:
                    return new Vector2(width - y, x);
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
#endif
    }
}
