using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

#if ENABLE_WINMD_SUPPORT && (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
using Windows.AI.MachineLearning;
#endif

#if USE_ONNXRUNTIME && !(ENABLE_WINMD_SUPPORT && (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR)
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
#endif

namespace HoloFaceRecognition
{
    public sealed class OnnxFaceRecognizer : IDisposable
    {
        const string DefaultModelFileName = "ghostfacenet.onnx";
        const int InputChannels = 3;

        public string ModelName { get; private set; } = "mock-recognizer";
        public int EmbeddingSize { get; private set; } = 512;
        public float LastInferenceMs { get; private set; }
        float[] _inputTensorBuffer;

        enum InputTensorLayout
        {
            Nchw,
            Nhwc
        }

#if ENABLE_WINMD_SUPPORT && (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
        LearningModel _winmlModel;
        LearningModelSession _winmlSession;
        LearningModelBinding _winmlBinding;
        string _inputName;
        string _outputName;
        InputTensorLayout _inputLayout;
        long[] _winmlTensorShape;
#elif USE_ONNXRUNTIME
        InferenceSession _session;
        string _inputName;
        string _outputName;
        InputTensorLayout _inputLayout;
        int[] _onnxTensorShape;
#endif

        public Task InitializeAsync(string modelPath, string modelName)
        {
            ModelName = string.IsNullOrEmpty(modelName) ? Path.GetFileNameWithoutExtension(modelPath) : modelName;

            string resolvedModelPath = string.IsNullOrEmpty(modelPath)
                ? Path.Combine(Application.streamingAssetsPath, DefaultModelFileName)
                : modelPath;
            if (!File.Exists(resolvedModelPath))
            {
                string message = "ONNX model file not found. Expected path: " + resolvedModelPath;
                UnityEngine.Debug.LogError(message);
                throw new FileNotFoundException(message, resolvedModelPath);
            }

#if ENABLE_WINMD_SUPPORT && (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
            _winmlModel = LearningModel.LoadFromFilePath(resolvedModelPath);
            _winmlSession = new LearningModelSession(_winmlModel);

            var input = _winmlModel.InputFeatures.FirstOrDefault() as TensorFeatureDescriptor;
            if (input == null || string.IsNullOrEmpty(input.Name))
            {
                UnityEngine.Debug.LogError("WinML model has no tensor input metadata.");
                throw new InvalidOperationException("WinML model has no tensor input metadata.");
            }

            var output = _winmlModel.OutputFeatures.FirstOrDefault() as TensorFeatureDescriptor;
            if (output == null || string.IsNullOrEmpty(output.Name))
            {
                UnityEngine.Debug.LogError("WinML model has no tensor output metadata.");
                throw new InvalidOperationException("WinML model has no tensor output metadata.");
            }

            _inputName = input.Name;
            _outputName = output.Name;

            long[] inputShape = ToShapeArray(input.Shape);
            long[] outputShape = ToShapeArray(output.Shape);

            if (!TryGetInputTensorLayout(inputShape, out _inputLayout))
            {
                string message = "WinML input shape is not supported. Expected [1, 3, 112, 112] NCHW or [1, 112, 112, 3] NHWC, actual: " + FormatShape(inputShape);
                UnityEngine.Debug.LogError(message);
                throw new InvalidOperationException(message);
            }

            if (outputShape != null && outputShape.Length > 0)
                EmbeddingSize = Math.Abs((int)outputShape[outputShape.Length - 1]);

            _winmlTensorShape = _inputLayout == InputTensorLayout.Nchw
                ? new long[] { 1, InputChannels, FaceAligner.OutputSize, FaceAligner.OutputSize }
                : new long[] { 1, FaceAligner.OutputSize, FaceAligner.OutputSize, InputChannels };
            _inputTensorBuffer = new float[FaceAligner.OutputSize * FaceAligner.OutputSize * InputChannels];
            _winmlBinding = new LearningModelBinding(_winmlSession);

            UnityEngine.Debug.Log(
                "WinML model loaded.\n" +
                "modelPath: " + resolvedModelPath + "\n" +
                "input name: " + _inputName + "\n" +
                "input shape: " + FormatShape(inputShape) + "\n" +
                "input layout: " + _inputLayout + "\n" +
                "output name: " + _outputName + "\n" +
                "output shape: " + FormatShape(outputShape) + "\n" +
                "embedding dimension: " + EmbeddingSize);
#elif USE_ONNXRUNTIME
            try
            {
                _session = new InferenceSession(resolvedModelPath);

                var input = _session.InputMetadata.FirstOrDefault();
                if (string.IsNullOrEmpty(input.Key))
                {
                    UnityEngine.Debug.LogError("ONNX model has no input metadata.");
                    throw new InvalidOperationException("ONNX model has no input metadata.");
                }

                var output = _session.OutputMetadata.FirstOrDefault();
                if (string.IsNullOrEmpty(output.Key))
                {
                    UnityEngine.Debug.LogError("ONNX model has no output metadata.");
                    throw new InvalidOperationException("ONNX model has no output metadata.");
                }

                _inputName = input.Key;
                _outputName = output.Key;

                long[] inputShape = ToShapeArray(input.Value.Dimensions);
                long[] outputShape = ToShapeArray(output.Value.Dimensions);

                if (!TryGetInputTensorLayout(inputShape, out _inputLayout))
                {
                    string message = "ONNX input shape is not supported. Expected [1, 3, 112, 112] NCHW or [1, 112, 112, 3] NHWC, actual: " + FormatShape(inputShape);
                    UnityEngine.Debug.LogError(message);
                    throw new InvalidOperationException(message);
                }

                if (outputShape != null && outputShape.Length > 0)
                    EmbeddingSize = Math.Abs((int)outputShape[outputShape.Length - 1]);

                _onnxTensorShape = _inputLayout == InputTensorLayout.Nchw
                    ? new[] { 1, InputChannels, FaceAligner.OutputSize, FaceAligner.OutputSize }
                    : new[] { 1, FaceAligner.OutputSize, FaceAligner.OutputSize, InputChannels };
                _inputTensorBuffer = new float[FaceAligner.OutputSize * FaceAligner.OutputSize * InputChannels];

                UnityEngine.Debug.Log(
                    "ONNX Runtime model loaded.\n" +
                    "modelPath: " + resolvedModelPath + "\n" +
                    "input name: " + _inputName + "\n" +
                    "input shape: " + FormatShape(inputShape) + "\n" +
                    "input layout: " + _inputLayout + "\n" +
                    "output name: " + _outputName + "\n" +
                    "output shape: " + FormatShape(outputShape) + "\n" +
                    "embedding dimension: " + EmbeddingSize);
            }
            catch (DllNotFoundException ex)
            {
                UnityEngine.Debug.LogError("ONNX Runtime native DLL is missing. Place onnxruntime.dll under Assets/Plugins/x86_64/. " + ex);
                throw;
            }
            catch (BadImageFormatException ex)
            {
                UnityEngine.Debug.LogError("ONNX Runtime DLL architecture mismatch. Use Windows x64 DLLs for Unity Editor Windows x64. " + ex);
                throw;
            }
            catch (TypeInitializationException ex)
            {
                UnityEngine.Debug.LogError("ONNX Runtime failed to initialize. Check Microsoft.ML.OnnxRuntime.dll and onnxruntime.dll plugin placement. " + ex);
                throw;
            }
#else
            UnityEngine.Debug.LogWarning("USE_ONNXRUNTIME is not defined. OnnxFaceRecognizer will return deterministic mock embeddings.");
#endif
            return Task.CompletedTask;
        }

        public Task<float[]> ExtractEmbeddingAsync(Color32[] alignedFace112)
        {
            return ExtractEmbeddingAsync(alignedFace112, FaceAligner.OutputSize, FaceAligner.OutputSize);
        }

        public Task<float[]> ExtractEmbeddingAsync(Color32[] faceCrop, int cropWidth, int cropHeight)
        {
#if ENABLE_WINMD_SUPPORT && (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
            return ExtractEmbeddingWithWinMLAsync(faceCrop, cropWidth, cropHeight);
#else
            return Task.Run(() => ExtractEmbedding(faceCrop, cropWidth, cropHeight));
#endif
        }

        public Task<float[]> ExtractEmbeddingAsync(Texture2D faceTexture)
        {
            if (faceTexture == null)
            {
                UnityEngine.Debug.LogError("Cannot extract embedding: face Texture2D is null.");
                throw new ArgumentNullException("faceTexture");
            }

            return ExtractEmbeddingAsync(faceTexture.GetPixels32(), faceTexture.width, faceTexture.height);
        }

        float[] ExtractEmbedding(Color32[] faceCrop, int cropWidth, int cropHeight)
        {
            var sw = Stopwatch.StartNew();

#if USE_ONNXRUNTIME && !(ENABLE_WINMD_SUPPORT && (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR)
            if (_session == null)
            {
                UnityEngine.Debug.LogError("Cannot extract embedding: ONNX Runtime session has not been initialized.");
                throw new InvalidOperationException("Recognizer has not been initialized.");
            }

            DenseTensor<float> input = Preprocess(faceCrop, cropWidth, cropHeight, _inputLayout);
            using (var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, input) }))
            {
                foreach (var result in results)
                {
                    float[] embedding = result.AsEnumerable<float>().ToArray();
                    LastInferenceMs = (float)sw.Elapsed.TotalMilliseconds;
                    return L2Normalize(embedding);
                }
            }
            throw new InvalidOperationException("ONNX model produced no output.");
#else
            float[] embedding = new float[EmbeddingSize];
            if (faceCrop != null)
            {
                for (int i = 0; i < faceCrop.Length; i++)
                {
                    Color32 c = faceCrop[i];
                    int bucket = (c.r * 3 + c.g * 5 + c.b * 7 + i) % embedding.Length;
                    embedding[bucket] += (c.r + c.g + c.b) / 765f;
                }
            }
            LastInferenceMs = (float)sw.Elapsed.TotalMilliseconds;
            return L2Normalize(embedding);
#endif
        }

#if ENABLE_WINMD_SUPPORT && (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
        async Task<float[]> ExtractEmbeddingWithWinMLAsync(Color32[] faceCrop, int cropWidth, int cropHeight)
        {
            var sw = Stopwatch.StartNew();

            if (_winmlSession == null)
            {
                UnityEngine.Debug.LogError("Cannot extract embedding: WinML session has not been initialized.");
                throw new InvalidOperationException("Recognizer has not been initialized.");
            }

            float[] inputData = PreprocessToArray(faceCrop, cropWidth, cropHeight, _inputLayout);
            TensorFloat inputTensor = TensorFloat.CreateFromArray(_winmlTensorShape, inputData);
            _winmlBinding.Bind(_inputName, inputTensor);

            LearningModelEvaluationResult result = await _winmlSession.EvaluateAsync(_winmlBinding, "face-embedding");
            TensorFloat outputTensor = result.Outputs[_outputName] as TensorFloat;
            if (outputTensor == null)
                throw new InvalidOperationException("WinML model output is not a TensorFloat.");

            float[] embedding = outputTensor.GetAsVectorView().ToArray();
            LastInferenceMs = (float)sw.Elapsed.TotalMilliseconds;
            return L2Normalize(embedding);
        }
#endif

#if USE_ONNXRUNTIME && !(ENABLE_WINMD_SUPPORT && (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR)
        DenseTensor<float> Preprocess(Color32[] pixels, int sourceWidth, int sourceHeight, InputTensorLayout inputLayout)
        {
            return new DenseTensor<float>(PreprocessToArray(pixels, sourceWidth, sourceHeight, inputLayout), _onnxTensorShape);
        }
#endif

        float[] PreprocessToArray(Color32[] pixels, int sourceWidth, int sourceHeight, InputTensorLayout inputLayout)
        {
            if (pixels == null || pixels.Length == 0)
            {
                UnityEngine.Debug.LogError("Cannot preprocess face crop: pixels are null or empty.");
                throw new ArgumentException("Face crop pixels are null or empty.", "pixels");
            }

            if (sourceWidth <= 0 || sourceHeight <= 0 || pixels.Length < sourceWidth * sourceHeight)
            {
                string message = "Cannot preprocess face crop: invalid crop size. width=" + sourceWidth + ", height=" + sourceHeight + ", pixels=" + pixels.Length;
                UnityEngine.Debug.LogError(message);
                throw new ArgumentException(message, "pixels");
            }

            if (_inputTensorBuffer == null || _inputTensorBuffer.Length != FaceAligner.OutputSize * FaceAligner.OutputSize * InputChannels)
                _inputTensorBuffer = new float[FaceAligner.OutputSize * FaceAligner.OutputSize * InputChannels];

            float[] tensor = _inputTensorBuffer;
            for (int y = 0; y < FaceAligner.OutputSize; y++)
            {
                float sourceY = (y + 0.5f) * sourceHeight / FaceAligner.OutputSize - 0.5f;
                for (int x = 0; x < FaceAligner.OutputSize; x++)
                {
                    float sourceX = (x + 0.5f) * sourceWidth / FaceAligner.OutputSize - 0.5f;
                    Color32 c = SampleBilinear(pixels, sourceWidth, sourceHeight, sourceX, sourceY);
                    if (inputLayout == InputTensorLayout.Nchw)
                    {
                        int planeSize = FaceAligner.OutputSize * FaceAligner.OutputSize;
                        int pixelIndex = y * FaceAligner.OutputSize + x;
                        tensor[pixelIndex] = (c.r - 127.5f) / 128f;
                        tensor[planeSize + pixelIndex] = (c.g - 127.5f) / 128f;
                        tensor[planeSize * 2 + pixelIndex] = (c.b - 127.5f) / 128f;
                    }
                    else
                    {
                        int index = (y * FaceAligner.OutputSize + x) * InputChannels;
                        tensor[index] = (c.r - 127.5f) / 128f;
                        tensor[index + 1] = (c.g - 127.5f) / 128f;
                        tensor[index + 2] = (c.b - 127.5f) / 128f;
                    }
                }
            }
            return tensor;
        }

        static bool TryGetInputTensorLayout(long[] shape, out InputTensorLayout inputLayout)
        {
            inputLayout = InputTensorLayout.Nchw;
            if (shape == null || shape.Length != 4 || (shape[0] != 1 && shape[0] >= 0))
                return false;

            if (shape[1] == InputChannels && shape[2] == FaceAligner.OutputSize && shape[3] == FaceAligner.OutputSize)
            {
                inputLayout = InputTensorLayout.Nchw;
                return true;
            }

            if (shape[1] == FaceAligner.OutputSize && shape[2] == FaceAligner.OutputSize && shape[3] == InputChannels)
            {
                inputLayout = InputTensorLayout.Nhwc;
                return true;
            }

            return false;
        }

        static long[] ToShapeArray(System.Collections.Generic.IEnumerable<int> shape)
        {
            return shape == null ? null : shape.Select(d => (long)d).ToArray();
        }

        static long[] ToShapeArray(System.Collections.Generic.IEnumerable<long> shape)
        {
            return shape == null ? null : shape.ToArray();
        }

        static string FormatShape(long[] shape)
        {
            return shape == null ? "[]" : "[" + string.Join(", ", shape.Select(d => d < 0 ? "dynamic" : d.ToString()).ToArray()) + "]";
        }

        static Color32 SampleBilinear(Color32[] pixels, int width, int height, float x, float y)
        {
            x = Mathf.Clamp(x, 0f, width - 1f);
            y = Mathf.Clamp(y, 0f, height - 1f);

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

        public static float[] L2Normalize(float[] vector)
        {
            double sum = 0;
            for (int i = 0; i < vector.Length; i++)
                sum += vector[i] * vector[i];

            float inv = sum > 1e-12 ? (float)(1.0 / Math.Sqrt(sum)) : 0f;
            for (int i = 0; i < vector.Length; i++)
                vector[i] *= inv;
            return vector;
        }

        public void Dispose()
        {
#if ENABLE_WINMD_SUPPORT && (UNITY_WSA || WINDOWS_UWP) && !UNITY_EDITOR
            _winmlSession?.Dispose();
            _winmlSession = null;
            _winmlModel?.Dispose();
            _winmlModel = null;
#elif USE_ONNXRUNTIME
            _session?.Dispose();
            _session = null;
#endif
        }
    }
}
