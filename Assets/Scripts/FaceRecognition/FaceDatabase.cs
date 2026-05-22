using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace HoloFaceRecognition
{
    [Serializable]
    public sealed class FaceEmbeddingRecord
    {
        public float[] embedding;
        public string createdAt;
        public string modelName;
    }

    [Serializable]
    public sealed class FacePersonRecord
    {
        public string name;
        public string personId;
        public List<FaceEmbeddingRecord> embeddings = new List<FaceEmbeddingRecord>();
    }

    [Serializable]
    public sealed class FaceDatabaseFile
    {
        public List<FacePersonRecord> people = new List<FacePersonRecord>();
    }

    [Serializable]
    sealed class LegacyFaceEmbeddingRecord
    {
        public string name;
        public string personId;
        public float[] embedding;
        public string createdAt;
        public string modelName;
    }

    [Serializable]
    sealed class LegacyFaceDatabaseFile
    {
        public List<LegacyFaceEmbeddingRecord> people = new List<LegacyFaceEmbeddingRecord>();
    }

    public sealed class FaceDatabase
    {
        const string RelativeDatabasePath = "FaceDB/embeddings.json";

        public FaceDatabaseFile Data { get; private set; } = new FaceDatabaseFile();
        public string RuntimePath { get; private set; }

        public async Task LoadAsync()
        {
            EnsureRuntimePath();

            if (!File.Exists(RuntimePath))
            {
                string seedJson = await ReadStreamingAssetsTextAsync(RelativeDatabasePath);
                File.WriteAllText(RuntimePath, string.IsNullOrWhiteSpace(seedJson) ? "{\"people\":[]}" : seedJson);
            }

            string json = File.ReadAllText(RuntimePath);
            Data = string.IsNullOrWhiteSpace(json) ? new FaceDatabaseFile() : JsonUtility.FromJson<FaceDatabaseFile>(json);
            if (Data == null || Data.people == null)
                Data = new FaceDatabaseFile();

            MigrateLegacyDataIfNeeded(json);
            NormalizeData();
        }

        public void AddEmbedding(string name, float[] embedding, string modelName)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Name is required.", nameof(name));
            if (embedding == null || embedding.Length == 0)
                throw new ArgumentException("Embedding is empty.", nameof(embedding));

            NormalizeData();

            string trimmedName = name.Trim();
            FacePersonRecord person = null;
            foreach (var item in Data.people)
            {
                if (string.Equals(item.name, trimmedName, StringComparison.OrdinalIgnoreCase))
                {
                    person = item;
                    break;
                }
            }

            if (person == null)
            {
                person = new FacePersonRecord
                {
                    name = trimmedName,
                    personId = Guid.NewGuid().ToString("N"),
                    embeddings = new List<FaceEmbeddingRecord>()
                };
                Data.people.Add(person);
            }

            if (person.embeddings == null)
                person.embeddings = new List<FaceEmbeddingRecord>();

            person.embeddings.Add(new FaceEmbeddingRecord
            {
                embedding = OnnxFaceRecognizer.L2Normalize((float[])embedding.Clone()),
                createdAt = DateTime.UtcNow.ToString("o"),
                modelName = modelName
            });
        }

        public void Clear()
        {
            if (Data == null)
                Data = new FaceDatabaseFile();

            if (Data.people == null)
                Data.people = new List<FacePersonRecord>();

            Data.people.Clear();
        }

        public void Save()
        {
            EnsureRuntimePath();
            Directory.CreateDirectory(Path.GetDirectoryName(RuntimePath));
            string json = JsonUtility.ToJson(Data, true);
            File.WriteAllText(RuntimePath, json);
        }

        void EnsureRuntimePath()
        {
            if (string.IsNullOrEmpty(RuntimePath))
                RuntimePath = Path.Combine(Application.persistentDataPath, RelativeDatabasePath);

            Directory.CreateDirectory(Path.GetDirectoryName(RuntimePath));
        }

        void NormalizeData()
        {
            if (Data == null)
                Data = new FaceDatabaseFile();

            if (Data.people == null)
                Data.people = new List<FacePersonRecord>();

            foreach (var person in Data.people)
            {
                if (string.IsNullOrWhiteSpace(person.personId))
                    person.personId = Guid.NewGuid().ToString("N");

                if (person.name != null)
                    person.name = person.name.Trim();

                if (person.embeddings == null)
                    person.embeddings = new List<FaceEmbeddingRecord>();
            }
        }

        void MigrateLegacyDataIfNeeded(string json)
        {
            if (string.IsNullOrWhiteSpace(json) || !DataLooksLegacy(json))
                return;

            LegacyFaceDatabaseFile legacy = JsonUtility.FromJson<LegacyFaceDatabaseFile>(json);
            var migrated = new FaceDatabaseFile();

            if (legacy != null && legacy.people != null)
            {
                foreach (var legacyRecord in legacy.people)
                {
                    if (legacyRecord == null || string.IsNullOrWhiteSpace(legacyRecord.name) || legacyRecord.embedding == null || legacyRecord.embedding.Length == 0)
                        continue;

                    string trimmedName = legacyRecord.name.Trim();
                    FacePersonRecord person = FindPerson(migrated.people, trimmedName);
                    if (person == null)
                    {
                        person = new FacePersonRecord
                        {
                            name = trimmedName,
                            personId = string.IsNullOrEmpty(legacyRecord.personId) ? Guid.NewGuid().ToString("N") : legacyRecord.personId,
                            embeddings = new List<FaceEmbeddingRecord>()
                        };
                        migrated.people.Add(person);
                    }

                    person.embeddings.Add(new FaceEmbeddingRecord
                    {
                        embedding = legacyRecord.embedding,
                        createdAt = legacyRecord.createdAt,
                        modelName = legacyRecord.modelName
                    });
                }
            }

            Data = migrated;
            Save();
        }

        bool DataLooksLegacy(string json)
        {
            if (json.IndexOf("\"embedding\"", StringComparison.Ordinal) >= 0 &&
                json.IndexOf("\"embeddings\"", StringComparison.Ordinal) < 0)
                return true;

            if (Data == null || Data.people == null || Data.people.Count == 0)
                return false;

            foreach (var person in Data.people)
            {
                if (person.embeddings == null)
                    return true;
            }

            return false;
        }

        static FacePersonRecord FindPerson(List<FacePersonRecord> people, string name)
        {
            foreach (var person in people)
            {
                if (string.Equals(person.name, name, StringComparison.OrdinalIgnoreCase))
                    return person;
            }

            return null;
        }

        static async Task<string> ReadStreamingAssetsTextAsync(string relativePath)
        {
            string path = CombineStreamingAssetsPath(relativePath);
            if (path.Contains("://") || path.Contains(":///"))
            {
                using (UnityWebRequest request = UnityWebRequest.Get(path))
                {
                    var op = request.SendWebRequest();
                    while (!op.isDone)
                        await Task.Yield();

#if UNITY_2020_2_OR_NEWER
                    if (request.result != UnityWebRequest.Result.Success)
#else
                    if (request.isNetworkError || request.isHttpError)
#endif
                        return null;

                    return request.downloadHandler.text;
                }
            }

            return File.Exists(path) ? File.ReadAllText(path) : null;
        }

        static string CombineStreamingAssetsPath(string relativePath)
        {
            string root = Application.streamingAssetsPath.TrimEnd('/', '\\');
            return root + "/" + relativePath.Replace("\\", "/");
        }
    }
}
