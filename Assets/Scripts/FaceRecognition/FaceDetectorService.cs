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
#if WINDOWS_UWP && !UNITY_EDITOR
        FaceDetector _detector;
#endif

        public async Task InitializeAsync()
        {
#if WINDOWS_UWP && !UNITY_EDITOR
            _detector = await FaceDetector.CreateAsync();
#else
            await Task.CompletedTask;
#endif
        }

        public async Task<List<FaceDetectionResult>> DetectAsync(CameraFrame frame)
        {
            var results = new List<FaceDetectionResult>();

            if (frame == null || frame.pixels == null || frame.width <= 0 || frame.height <= 0)
                return results;

#if WINDOWS_UWP && !UNITY_EDITOR
            if (_detector == null)
                await InitializeAsync();

            using (SoftwareBitmap bitmap = CreateSoftwareBitmap(frame))
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
#else
            await Task.CompletedTask;

            // Unity Editor / PC fallback:
            // Return a mock face box in the center of the frame.
            // This is only for testing the pipeline before deploying to HoloLens.
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
#endif

            return results;
        }

#if WINDOWS_UWP && !UNITY_EDITOR
        static SoftwareBitmap CreateSoftwareBitmap(CameraFrame frame)
        {
            var bitmap = new SoftwareBitmap(
                BitmapPixelFormat.Bgra8,
                frame.width,
                frame.height,
                BitmapAlphaMode.Premultiplied
            );

            byte[] bytes = new byte[frame.width * frame.height * 4];
            int length = Math.Min(frame.pixels.Length, frame.width * frame.height);

            for (int i = 0; i < length; i++)
            {
                Color32 c = frame.pixels[i];
                int o = i * 4;
                bytes[o + 0] = c.b;
                bytes[o + 1] = c.g;
                bytes[o + 2] = c.r;
                bytes[o + 3] = 255;
            }

            bitmap.CopyFromBuffer(bytes.AsBuffer());
            return bitmap;
        }
#endif
    }
}