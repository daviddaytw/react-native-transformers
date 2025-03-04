import React from "react";
import {
  StyleSheet,
  Text,
  Button,
  TextInput,
  SafeAreaView,
} from "react-native";
import * as FileSystem from "expo-file-system";
import { Pipeline } from "react-native-transformers";
import presets from "./presets.json";

export default function App() {
  const [progress, setProgress] = React.useState<number>();
  const [input, setInput] = React.useState<string>("We love local LLM");
  const [output, setOutput] = React.useState<string>();

  const loadModel = async (preset: {
    name: string;
    model: string;
    onnx_path: string;
    options?: any;
  }) => {
    console.log("loading");
    await Pipeline.TextGeneration.init(preset.model, preset.onnx_path, {
      verbose: true,
      fetch: async (url) => {
        try {
          console.log("Checking file... " + url);
          const fileName = url.split("/").pop()!;
          const localPath = FileSystem.documentDirectory + fileName;
      
          // Check if the file already exists
          const fileInfo = await FileSystem.getInfoAsync(localPath);
          if (fileInfo.exists) {
            console.log("File already exists: " + localPath);
            return localPath;
          }
      
          console.log("Downloading... " + url);
          const downloadResumable = FileSystem.createDownloadResumable(
            url,
            localPath,
            {},
            ({ totalBytesWritten, totalBytesExpectedToWrite }) => {
              setProgress(totalBytesWritten / totalBytesExpectedToWrite);
            }
          );
      
          const result = await downloadResumable.downloadAsync();
          if (!result) {
            throw new Error("Download failed.");
          }
      
          console.log("Downloaded to: " + result.uri);
          return result.uri;
        } catch (error) {
          console.error("Download error:", error);
          return null;
        }
      },
      ...preset.options,
    });
    console.log("loaded");
  };

  const AutoComplete = () => {
    Pipeline.TextGeneration.generate(input, setOutput);
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text>Select a model</Text>
      {presets.map((preset) => (
        <Button
          key={preset.name}
          title={preset.name}
          onPress={() => {
            loadModel(preset);
          }}
        />
      ))}
      <Text>Input: </Text>
      <TextInput value={input} onChangeText={setInput} style={styles.input} />
      <Text>Output: {output}</Text>
      <Text>{progress}</Text>
      <Button title="Run" onPress={AutoComplete} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  input: {
    borderWidth: 1,
    borderColor: "black",
  },
});
