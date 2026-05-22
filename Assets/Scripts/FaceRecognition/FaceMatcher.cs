using System;

namespace HoloFaceRecognition
{
    public sealed class FaceMatchResult
    {
        public string name = "Unknown";
        public string personId;
        public float similarity;
        public bool isKnown;
    }

    public sealed class FaceMatcher
    {
        public float Threshold { get; set; } = 0.5f;

        public FaceMatchResult Match(float[] embedding, FaceDatabase database)
        {
            var result = new FaceMatchResult();
            if (embedding == null || database?.Data?.people == null)
                return result;

            foreach (var person in database.Data.people)
            {
                if (person.embeddings == null)
                    continue;

                foreach (var record in person.embeddings)
                {
                    if (record.embedding == null || record.embedding.Length != embedding.Length)
                        continue;

                    float sim = CosineSimilarity(embedding, record.embedding);
                    if (sim > result.similarity)
                    {
                        result.similarity = sim;
                        result.name = person.name;
                        result.personId = person.personId;
                    }
                }
            }

            result.isKnown = result.similarity >= Threshold;
            if (!result.isKnown)
                result.name = "Unknown";
            return result;
        }

        public static float CosineSimilarity(float[] a, float[] b)
        {
            int count = Math.Min(a.Length, b.Length);
            double dot = 0;
            double aa = 0;
            double bb = 0;

            for (int i = 0; i < count; i++)
            {
                dot += a[i] * b[i];
                aa += a[i] * a[i];
                bb += b[i] * b[i];
            }

            double denom = Math.Sqrt(aa) * Math.Sqrt(bb);
            return denom > 1e-12 ? (float)(dot / denom) : 0f;
        }
    }
}
