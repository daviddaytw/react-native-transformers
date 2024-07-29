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
        console.log("downloading... " + url);
        const localpath = FileSystem.cacheDirectory + url.split("/").pop()!;

        const downloadResumable = FileSystem.createDownloadResumable(
          url,
          localpath,
          {},
          ({ totalBytesWritten, totalBytesExpectedToWrite }) => {
            setProgress(totalBytesWritten / totalBytesExpectedToWrite);
          },
        );
        const result = await downloadResumable.downloadAsync();
        if (result === undefined) {
          throw new Error("Download failed.");
        }
        console.log("downloaded as " + result.uri);
        return result.uri;
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
