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
            }
            catch (Exception ex)
            {
                LastStatus = "UWP FaceDetector init failed";
                LastError = ex.GetType().Name + ": " + ex.Message;
                UnityEngine.Debug.LogException(ex);
                throw;
            }
#else
            LastStatus = "Editor mock detector initialized";
            LastError = string.Empty;
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

                using (SoftwareBitmap bitmap = CreateGray8SoftwareBitmap(frame))
                {
                    var faces = await _detector.DetectFacesAsync(bitmap);
                    foreach (var face in faces)
                    {
                        BitmapBounds box = face.FaceBox;
                        results.Add(new FaceDetectionResult
                        {
                            pixelRect = new Rect(box.X, box.Y, box.Width, box.Height),
                            landmarks = null,
                            confidence = 1f
                        });
                    }
                }

                LastDetectedFaceCount = results.Count;
                LastStatus = "UWP detector faces=" + LastDetectedFaceCount;
                LastError = string.Empty;
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
            LastStatus = "Mock detector faces=" + LastDetectedFaceCount;
            LastError = string.Empty;
#endif

            return results;
        }

#if WINDOWS_UWP && !UNITY_EDITOR
        static SoftwareBitmap CreateGray8SoftwareBitmap(CameraFrame frame)
        {
            var bitmap = new SoftwareBitmap(
                BitmapPixelFormat.Gray8,
                frame.width,
                frame.height,
                BitmapAlphaMode.Ignore
            );

            byte[] bytes = new byte[frame.width * frame.height];
            int length = Math.Min(frame.pixels.Length, frame.width * frame.height);

            for (int i = 0; i < length; i++)
            {
                Color32 c = frame.pixels[i];
                bytes[i] = (byte)((c.r * 299 + c.g * 587 + c.b * 114) / 1000);
            }

            bitmap.CopyFromBuffer(bytes.AsBuffer());
            return bitmap;
        }
#endif
    }
}
