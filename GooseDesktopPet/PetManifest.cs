using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace GooseDesktopPet;

public sealed class PetManifest
{
    [JsonPropertyName("target_fps")]
    public double TargetFps { get; set; } = 12;

    [JsonPropertyName("preserve_y_states")]
    public List<string> PreserveYStates { get; set; } = [];

    [JsonPropertyName("states")]
    public List<PetStateManifest> States { get; set; } = [];

    public static PetManifest Load(string path)
    {
        var json = File.ReadAllText(path);
        var manifest = JsonSerializer.Deserialize<PetManifest>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        });

        return manifest ?? throw new InvalidOperationException($"Could not read pet manifest: {path}");
    }
}

public sealed class PetStateManifest
{
    [JsonPropertyName("state")]
    public string State { get; set; } = "";

    [JsonPropertyName("frames")]
    public List<string> Frames { get; set; } = [];
}
