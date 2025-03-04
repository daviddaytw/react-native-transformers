# react-native-transformers

![NPM Version](https://img.shields.io/npm/v/react-native-transformers)
[![codecov](https://codecov.io/github/daviddaytw/react-native-transformers/graph/badge.svg?token=G3D0Y33SI4)](https://codecov.io/github/daviddaytw/react-native-transformers)
[![TypeDoc](https://github.com/daviddaytw/react-native-transformers/actions/workflows/docs.yml/badge.svg)](https://daviddaytw.github.io/react-native-transformers)

**Run Hugging Face transformer models directly on your React Native and Expo applications with on-device inference. No cloud service required!**

## Overview

`react-native-transformers` empowers your mobile applications with AI capabilities by running transformer models directly on the device. This means your app can generate text, answer questions, and process language without sending data to external servers - enhancing privacy, reducing latency, and enabling offline functionality.

Built on top of ONNX Runtime, this library provides a streamlined API for integrating state-of-the-art language models into your React Native and Expo applications with minimal configuration.

## Key Features

- **On-device inference**: Run AI models locally without requiring an internet connection
- **Privacy-focused**: Keep user data on the device without sending it to external servers
- **Optimized performance**: Leverages ONNX Runtime for efficient model execution on mobile CPUs
- **Simple API**: Easy-to-use interface for model loading and inference
- **Expo compatibility**: Works seamlessly with both Expo managed and bare workflows

## Installation

### 1. Install peer dependencies

```sh
npm install onnxruntime-react-native
```

### 2. Install react-native-transformers

```sh
# React-Native
npm install react-native-transformers

# Expo
npx expo install react-native-transformers
```

### 3. Platform Configuration

<details>
  <summary><b>React Native CLI</b></summary>

Link the `onnxruntime-react-native` library:

```sh
npx react-native link onnxruntime-react-native
```
</details>

<details>
  <summary><b>Expo</b></summary>

Add the Expo plugin configuration in `app.json` or `app.config.js`:

```json
{
  "expo": {
    "plugins": [
      "onnxruntime-react-native"
    ]
  }
}
```
</details>

### 4. Babel Configuration

Add the `babel-plugin-transform-import-meta` plugin to your Babel configuration:

```js
// babel.config.js
module.exports = {
  // ... your existing config
  plugins: [
    // ... your existing plugins
    "babel-plugin-transform-import-meta"
  ]
};
```

You can follow this [document](https://docs.expo.dev/versions/latest/config/babel/) to create config file, and you need to run `npx expo start --clear` to clear the Metro bundler cache.

## Usage

### Text Generation

```javascript
import React, { useState, useEffect } from "react";
import { View, Text, Button } from "react-native";
import { Pipeline } from "react-native-transformers";

export default function App() {
  const [output, setOutput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);

  // Load model on component mount
  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    setIsLoading(true);
    try {
      // Load a small Llama model
      await Pipeline.TextGeneration.init(
        "Felladrin/onnx-Llama-160M-Chat-v1",
        "onnx/decoder_model_merged.onnx",
        {
          // The fetch function is required to download model files
          fetch: async (url) => {
            // In a real app, you might want to cache the downloaded files
            const response = await fetch(url);
            return response.url;
          }
        }
      );
      setIsModelReady(true);
    } catch (error) {
      console.error("Error loading model:", error);
      alert("Failed to load model: " + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const generateText = () => {
    setOutput("");
    // Generate text from the prompt and update the UI as tokens are generated
    Pipeline.TextGeneration.generate(
      "Write a short poem about programming:",
      (text) => setOutput(text)
    );
  };

  return (
    <View style={{ padding: 20 }}>
      <Button
        title={isModelReady ? "Generate Text" : "Load Model"}
        onPress={isModelReady ? generateText : loadModel}
        disabled={isLoading}
      />
      <Text style={{ marginTop: 20 }}>
        {output || "Generated text will appear here"}
      </Text>
    </View>
  );
}
```

### With Custom Model Download

For Expo applications, use `expo-file-system` to download models with progress tracking:

```javascript
import * as FileSystem from "expo-file-system";
import { Pipeline } from "react-native-transformers";

// In your model loading function
await Pipeline.TextGeneration.init("model-repo", "model-file", {
  fetch: async (url) => {
    const localPath = FileSystem.cacheDirectory + url.split("/").pop();

    // Check if file already exists
    const fileInfo = await FileSystem.getInfoAsync(localPath);
    if (fileInfo.exists) {
      console.log("Model already downloaded, using cached version");
      return localPath;
    }

    // Download file with progress tracking
    const downloadResumable = FileSystem.createDownloadResumable(
      url,
      localPath,
      {},
      (progress) => {
        const percentComplete = progress.totalBytesWritten / progress.totalBytesExpectedToWrite;
        console.log(`Download progress: ${(percentComplete * 100).toFixed(1)}%`);
      }
    );

    const result = await downloadResumable.downloadAsync();
    return result?.uri;
  }
});
```

## Supported Models

`react-native-transformers` works with ONNX-formatted models from Hugging Face. Here are some recommended models based on size and performance:

| Model | Type | Size | Description |
|-------|------|------|-------------|
| [Felladrin/onnx-Llama-160M-Chat-v1](https://huggingface.co/Felladrin/onnx-Llama-160M-Chat-v1) | Text Generation | ~300MB | Small Llama model (160M parameters) |
| [microsoft/Phi-3-mini-4k-instruct-onnx-web](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web) | Text Generation | ~1.5GB | Microsoft's Phi-3-mini model |
| [Xenova/distilgpt2_onnx-quantized](https://huggingface.co/Xenova/distilgpt2_onnx-quantized) | Text Generation | ~165MB | Quantized DistilGPT-2 |
| [Xenova/tiny-mamba-onnx](https://huggingface.co/Xenova/tiny-mamba-onnx) | Text Generation | ~85MB | Tiny Mamba model |
| [Xenova/all-MiniLM-L6-v2-onnx](https://huggingface.co/Xenova/all-MiniLM-L6-v2-onnx) | Text Embedding | ~80MB | Sentence embedding model |

## API Reference

For detailed API documentation, please visit our [TypeDoc documentation](https://daviddaytw.github.io/react-native-transformers/).

## Contributing

Contributions are welcome! See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [ONNX Runtime](https://onnxruntime.ai/) for efficient model execution on mobile devices
- [@xenova/transformers](https://www.npmjs.com/package/@xenova/transformers) for transformer model implementations
- [Hugging Face](https://huggingface.co/) for providing pre-trained models and model hosting

## External Links

- [Expo Plugins Documentation](https://docs.expo.dev/guides/config-plugins/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [ONNX Format Documentation](https://onnx.ai/onnx/intro/)
