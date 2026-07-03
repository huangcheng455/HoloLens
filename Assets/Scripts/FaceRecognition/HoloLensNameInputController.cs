using TMPro;
using Microsoft.MixedReality.Toolkit.UI;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace HoloFaceRecognition
{
    public sealed class HoloLensNameInputController : MonoBehaviour
    {
        [Header("Target UI")]
        public InputField targetInputField;
        public TMP_Text nameButtonLabel;
        public Text statusText;

        [Header("Recognition")]
        public FaceRecognitionManager faceRecognitionManager;

        [Header("Display")]
        public string emptyNameLabel = "Name: Tap to edit";
        public string nameLabelPrefix = "Name: ";

        [Header("Keyboard")]
        public string keyboardPlaceholder = "Enter name";

        private TouchScreenKeyboard _keyboard;
        private string _nameBeforeEditing = string.Empty;

        private void Awake()
        {
            if (faceRecognitionManager == null)
            {
                faceRecognitionManager = GetComponent<FaceRecognitionManager>();
            }

            EnsureLegacyInputClickTarget();
            BindNameInputButtons();
            RefreshNameButtonLabel();
        }

        private void Start()
        {
            BindNameInputButtons();
            RefreshNameButtonLabel();
        }

        /// <summary>
        /// Kept for compatibility with any old button binding.
        /// Registration itself is still performed by the Register button.
        /// </summary>
        public void OpenKeyboardThenRegister()
        {
            RegisterNow();
        }

        /// <summary>
        /// Bind the MRTK name button's OnClick event to this method.
        /// </summary>
        public void OpenKeyboardForNameInput()
        {
            if (_keyboard != null && _keyboard.status == TouchScreenKeyboard.Status.Visible)
            {
                return;
            }

            _nameBeforeEditing = GetCurrentName();
            string initialText = _nameBeforeEditing;

#if UNITY_EDITOR
            if (targetInputField != null)
            {
                targetInputField.gameObject.SetActive(true);
                targetInputField.text = initialText;
                targetInputField.Select();
                targetInputField.ActivateInputField();
            }

            SetStatus("Enter name. The HoloLens system keyboard appears only on the device.");
#else
            _keyboard = TouchScreenKeyboard.Open(
                initialText,
                TouchScreenKeyboardType.Default,
                false, // autocorrection
                false, // multiline
                false, // secure
                false, // alert
                keyboardPlaceholder
            );

            SetStatus("Enter name.");
#endif
        }

        private void Update()
        {
            if (_keyboard == null)
            {
                return;
            }

            if (_keyboard.status == TouchScreenKeyboard.Status.Visible)
            {
                ApplyName(_keyboard.text);
                return;
            }

            if (_keyboard.status == TouchScreenKeyboard.Status.Done)
            {
                string finalName = string.IsNullOrWhiteSpace(_keyboard.text)
                    ? string.Empty
                    : _keyboard.text.Trim();

                ApplyName(finalName);
                _keyboard = null;
                SetStatus(string.IsNullOrEmpty(finalName)
                    ? "Name cleared."
                    : "Name: " + finalName);
                return;
            }

            if (_keyboard.status == TouchScreenKeyboard.Status.Canceled ||
                _keyboard.status == TouchScreenKeyboard.Status.LostFocus)
            {
                ApplyName(_nameBeforeEditing);
                _keyboard = null;
                SetStatus("Name input canceled.");
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

            string currentName = GetCurrentName();
            if (string.IsNullOrEmpty(currentName))
            {
                SetStatus("Please enter a name before registering.");
                return;
            }

            SetStatus("Registering: " + currentName);
            faceRecognitionManager.RegisterCurrentFaceFromUI();
        }

        public bool UsesInputField(InputField inputField)
        {
            return targetInputField != null && targetInputField == inputField;
        }

        public void ClearNameAfterSuccessfulRegister()
        {
            ApplyName(string.Empty);
        }

        private void ApplyName(string value)
        {
            if (targetInputField != null)
            {
                targetInputField.text = value ?? string.Empty;
            }

            RefreshNameButtonLabel();
        }

        private string GetCurrentName()
        {
            if (targetInputField == null || string.IsNullOrWhiteSpace(targetInputField.text))
            {
                return string.Empty;
            }

            return targetInputField.text.Trim();
        }

        private void RefreshNameButtonLabel()
        {
            if (nameButtonLabel == null)
            {
                return;
            }

            string currentName = GetCurrentName();
            nameButtonLabel.text = string.IsNullOrEmpty(currentName)
                ? emptyNameLabel
                : nameLabelPrefix + currentName;
        }

        private void EnsureLegacyInputClickTarget()
        {
            if (targetInputField == null)
            {
                return;
            }

            HoloLensNameInputClickTarget clickTarget =
                targetInputField.GetComponent<HoloLensNameInputClickTarget>();

            if (clickTarget == null)
            {
                clickTarget = targetInputField.gameObject.AddComponent<HoloLensNameInputClickTarget>();
            }

            clickTarget.owner = this;
        }

        private void BindNameInputButtons()
        {
            Interactable[] interactables = FindObjectsOfType<Interactable>();
            for (int i = 0; i < interactables.Length; i++)
            {
                Interactable interactable = interactables[i];
                if (interactable != null && IsNameInputButton(interactable.gameObject))
                {
                    interactable.OnClick.RemoveListener(OpenKeyboardForNameInput);
                    interactable.OnClick.AddListener(OpenKeyboardForNameInput);
                    CacheNameButtonLabel(interactable.gameObject);
                }
            }

            Button[] buttons = FindObjectsOfType<Button>();
            for (int i = 0; i < buttons.Length; i++)
            {
                Button button = buttons[i];
                if (button != null && IsNameInputButton(button.gameObject))
                {
                    button.onClick.RemoveListener(OpenKeyboardForNameInput);
                    button.onClick.AddListener(OpenKeyboardForNameInput);
                }
            }
        }

        private bool IsNameInputButton(GameObject candidate)
        {
            if (candidate == null)
            {
                return false;
            }

            string text = (candidate.name + " " + GetChildLabelText(candidate)).ToLowerInvariant();
            return (text.Contains("input name") ||
                    text.Contains("inputname") ||
                    text.Contains("name input") ||
                    text.Contains("nameinput")) &&
                   !text.Contains("register") &&
                   !text.Contains("clear");
        }

        private string GetChildLabelText(GameObject candidate)
        {
            TMP_Text tmpLabel = candidate.GetComponentInChildren<TMP_Text>(true);
            if (tmpLabel != null)
            {
                return tmpLabel.text;
            }

            Text uiLabel = candidate.GetComponentInChildren<Text>(true);
            return uiLabel == null ? string.Empty : uiLabel.text;
        }

        private void CacheNameButtonLabel(GameObject candidate)
        {
            if (nameButtonLabel != null || candidate == null)
            {
                return;
            }

            nameButtonLabel = candidate.GetComponentInChildren<TMP_Text>(true);
        }

        private void SetStatus(string message)
        {
            if (statusText != null)
            {
                statusText.text = message;
            }
        }
    }

    /// <summary>
    /// Optional fallback for clicking the legacy Unity UI InputField.
    /// The recommended HoloLens interaction is the MRTK name button.
    /// </summary>
    public sealed class HoloLensNameInputClickTarget : MonoBehaviour, IPointerClickHandler
    {
        public HoloLensNameInputController owner;

        public void OnPointerClick(PointerEventData eventData)
        {
            if (owner != null)
            {
                owner.OpenKeyboardForNameInput();
            }
        }
    }
}
