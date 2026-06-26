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

#if WINDOWS_UWP && !UNITY_EDITOR
        FaceDetector _detector;
#endif

        public async Task InitializeAsync()
        {
#if WINDOWS_UWP && !UNITY_EDITOR
            try
            {
                _detector = await FaceDetector.CreateAsync();
                LastStatus = "UWP FaceDetector initialized";
                LastError = string.Empty;
                UnityEngine.Debug.Log(LastStatus);
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
            UnityEngine.Debug.Log(LastStatus);
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
                IList<DetectedFace> detectedFaces = null;
                foreach (int rotation in BuildRotationAttempts(frame.rotationAngle))
                {
                    using (SoftwareBitmap bitmap = CreateGray8SoftwareBitmap(frame, rotation))
                    {
                        detectedFaces = await _detector.DetectFacesAsync(bitmap);
                        rotationUsed = rotation;
                        if (detectedFaces != null && detectedFaces.Count > 0)
                            break;
                    }
                }

                if (detectedFaces != null)
                {
                    foreach (var face in detectedFaces)
                    {
                        BitmapBounds box = face.FaceBox;
                        Rect pixelRect = MapDetectedRectToFrame(
                            new Rect(box.X, box.Y, box.Width, box.Height),
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
                LastStatus = "faces detected: " + LastDetectedFaceCount + " rotation=" + rotationUsed;
                LastError = string.Empty;
                UnityEngine.Debug.Log(LastStatus);
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
            UnityEngine.Debug.Log(LastStatus);
#endif

            return results;
        }

#if WINDOWS_UWP && !UNITY_EDITOR
        static int[] BuildRotationAttempts(int preferredRotation)
        {
            int normalized = NormalizeRotation(preferredRotation);
            var rotations = new List<int> { normalized, 0, 90, 270, 180 };
            for (int i = rotations.Count - 1; i >= 0; i--)
            {
                if (rotations.IndexOf(rotations[i]) != i)
                    rotations.RemoveAt(i);
            }

            return rotations.ToArray();
        }

        static SoftwareBitmap CreateGray8SoftwareBitmap(CameraFrame frame, int rotation)
        {
            int normalizedRotation = NormalizeRotation(rotation);
            int bitmapWidth = normalizedRotation == 90 || normalizedRotation == 270 ? frame.height : frame.width;
            int bitmapHeight = normalizedRotation == 90 || normalizedRotation == 270 ? frame.width : frame.height;
            var bitmap = new SoftwareBitmap(
                BitmapPixelFormat.Gray8,
                bitmapWidth,
                bitmapHeight,
                BitmapAlphaMode.Ignore
            );

            byte[] bytes = new byte[bitmapWidth * bitmapHeight];
            int length = Math.Min(frame.pixels.Length, frame.width * frame.height);

            for (int i = 0; i < length; i++)
            {
                int sourceX = i % frame.width;
                int sourceY = i / frame.width;
                int targetX;
                int targetY;
                MapFramePointToRotated(sourceX, sourceY, frame.width, frame.height, normalizedRotation, out targetX, out targetY);

                Color32 c = frame.pixels[i];
                bytes[targetY * bitmapWidth + targetX] = (byte)((c.r * 299 + c.g * 587 + c.b * 114) / 1000);
            }

            bitmap.CopyFromBuffer(bytes.AsBuffer());
            return bitmap;
        }

        static void MapFramePointToRotated(int x, int y, int width, int height, int rotation, out int targetX, out int targetY)
        {
            switch (rotation)
            {
                case 90:
                    targetX = height - 1 - y;
                    targetY = x;
                    break;
                case 180:
                    targetX = width - 1 - x;
                    targetY = height - 1 - y;
                    break;
                case 270:
                    targetX = y;
                    targetY = width - 1 - x;
                    break;
                default:
                    targetX = x;
                    targetY = y;
                    break;
            }
        }

        static Rect MapDetectedRectToFrame(Rect rotatedRect, int frameWidth, int frameHeight, int rotation)
        {
            int normalizedRotation = NormalizeRotation(rotation);
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
