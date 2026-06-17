using UnityEditor;
using UnityEngine;

public static class AddMainSceneToBuild
{
    [MenuItem("Tools/Add MainScene To Build Settings")]
    public static void AddScene()
    {
        string scenePath = "Assets/Scenes/MainScene.unity";

        EditorBuildSettings.scenes = new[]
        {
            new EditorBuildSettingsScene(scenePath, true)
        };

        Debug.Log("Added MainScene to Build Settings: " + scenePath);
    }
}