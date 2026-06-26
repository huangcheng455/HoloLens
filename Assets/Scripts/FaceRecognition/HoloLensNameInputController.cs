using UnityEngine;
using UnityEngine.UI;

namespace HoloFaceRecognition
{
    public sealed class HoloLensNameInputController : MonoBehaviour
    {
        [Header("Target UI")]
        public InputField targetInputField;
        public Text statusText;

        [Header("Recognition")]
        public FaceRecognitionManager faceRecognitionManager;

        [Header("Defaults")]
        public string fallbackName = "HoloLensUser";
        public string keyboardPlaceholder = "Enter name";

        private TouchScreenKeyboard _keyboard;
        private bool _registerAfterKeyboard;

        private void Awake()
        {
            if (faceRecognitionManager == null)
            {
                faceRecognitionManager = GetComponent<FaceRecognitionManager>();
            }
        }

        /// <summary>
        /// MRTK Register 按钮调用这个方法：
        /// 有姓名时直接注册；没有姓名时打开键盘，输入完成后自动注册当前人脸。
        /// </summary>
        public void OpenKeyboardThenRegister()
        {
            string currentName = GetCurrentName();
            if (!string.IsNullOrWhiteSpace(currentName))
            {
                ApplyName(currentName);
                SetStatus("Registering: " + currentName);
                RegisterNow();
                return;
            }

            _registerAfterKeyboard = true;
            string initialText = GetFallbackName();

#if UNITY_EDITOR
            if (targetInputField != null)
            {
                targetInputField.text = initialText;
                targetInputField.ActivateInputField();
                targetInputField.Select();
            }

            SetStatus("Editor: type name in input field, then test registration on device.");
            return;
#else
            _keyboard = TouchScreenKeyboard.Open(
                initialText,
                TouchScreenKeyboardType.Default,
                false,
                false,
                false,
                false,
                keyboardPlaceholder
            );

            SetStatus("Enter name, then confirm to register.");
#endif
        }

        private void Update()
        {
            if (_keyboard == null)
                return;

            if (_keyboard.status == TouchScreenKeyboard.Status.Visible)
            {
                ApplyName(_keyboard.text);
                return;
            }

            if (_keyboard.status == TouchScreenKeyboard.Status.Done)
            {
                ApplyName(_keyboard.text);

                string finalName = GetCurrentName();
                ApplyName(finalName);

                SetStatus("Registering: " + finalName);

                _keyboard = null;

                if (_registerAfterKeyboard)
                {
                    _registerAfterKeyboard = false;
                    RegisterNow();
                }

                return;
            }

            if (_keyboard.status == TouchScreenKeyboard.Status.Canceled ||
                _keyboard.status == TouchScreenKeyboard.Status.LostFocus)
            {
                _keyboard = null;
                _registerAfterKeyboard = false;
                SetStatus("Name input canceled. Registration skipped.");
            }
        }

        private void RegisterNow()
        {
            if (faceRecognitionManager == null)
            {
                SetStatus("Register failed: FaceRecognitionManager is missing.");
                Debug.LogError("FaceRecognitionManager is missing.");
                return;
            }

            faceRecognitionManager.RegisterCurrentFaceForHoloLensTest();
        }

        private void ApplyName(string value)
        {
            if (targetInputField != null)
            {
                targetInputField.text = value ?? string.Empty;
            }
        }

        private string GetCurrentName()
        {
            if (targetInputField == null)
                return string.Empty;

            if (string.IsNullOrWhiteSpace(targetInputField.text))
                return string.Empty;

            return targetInputField.text.Trim();
        }

        private string GetFallbackName()
        {
            return string.IsNullOrWhiteSpace(fallbackName) ? "HoloLensUser" : fallbackName.Trim();
        }

        private void SetStatus(string message)
        {
            Debug.Log(message);

            if (statusText != null)
            {
                statusText.text = message;
            }
        }
    }
}
