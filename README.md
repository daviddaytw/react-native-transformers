# react-native-transformers

![NPM Version](https://img.shields.io/npm/v/react-native-transformers)
[![codecov](https://codecov.io/github/daviddaytw/react-native-transformers/graph/badge.svg?token=G3D0Y33SI4)](https://codecov.io/github/daviddaytw/react-native-transformers)
[![TypeDoc](https://github.com/daviddaytw/react-native-transformers/actions/workflows/docs.yml/badge.svg)](https://daviddaytw.github.io/react-native-transformers)

`react-native-transformers` is a React Native library for running Large Language Models (LLMs) from Hugging Face on your mobile applications locally. It supports both iOS and Android platforms, allowing you to leverage advanced AI models directly on your device without requiring an internet connection.

## Features

- On-device transformer model support for both text generation and text embedding
- Local inference without internet connectivity
- Compatible with iOS and Android platforms
- Simple API for model loading and inference
- Support for Hugging Face models in ONNX format
- Built on top of ONNX Runtime for efficient model execution
- TypeScript support with full type definitions

## Installation

To use `react-native-transformers`, you need to install `onnxruntime-react-native` as a peer dependency. Follow the steps below:

### 1. Install the peer dependency:

   ```sh
   npm install onnxruntime-react-native
   ```

### 2. Install `react-native-transformers`:

   ```sh
   npm install react-native-transformers
   ```

### 3. Configure React-Native or Expo

<details>
  <summary>React Native CLI</summary>

  - Link the `onnxruntime-react-native` library:

    ```sh
    npx react-native link onnxruntime-react-native
    ```
</details>

<details>
  <summary>Expo</summary>

  - Install the Expo plugin configuration in `app.json` or `app.config.js`:

    ```json
    {
      "expo": {
        "plugins": [
          "onnxruntime-react-native"
        ],
      }
    }
    ```
</details>

### 4. Babel Configuration

  You need to add the `babel-plugin-transform-import-meta` plugin to your Babel configuration (e.g., `.babelrc` or `babel.config.js`):

   ```json
   {
     "plugins": ["babel-plugin-transform-import-meta"]
   }
   ```

## Usage

### Text Generation Example

```javascript
import React from "react";
import { View, Text, Button } from "react-native";
import { Pipeline } from "react-native-transformers";

export default function App() {
  const [output, setOutput] = React.useState("");

  // Function to initialize the model
  const loadModel = async () => {
    await Pipeline.TextGeneration.init("Felladrin/onnx-Llama-160M-Chat-v1", "onnx/decoder_model_merged.onnx");
  };

  // Function to generate text
  const generateText = () => {
    Pipeline.TextGeneration.generate("Hello world", setOutput);
  };

  return (
    <View>
      <Button title="Load Model" onPress={loadModel} />
      <Button title="Generate Text" onPress={generateText} />
      <Text>Output: {output}</Text>
    </View>
  );
}
```

### Text Embedding Example

```javascript
import React from "react";
import { View, Text, Button } from "react-native";
import { Pipeline } from "react-native-transformers";

export default function App() {
  const [embedding, setEmbedding] = React.useState([]);

  // Function to initialize the model
  const loadModel = async () => {
    await Pipeline.TextEmbedding.init("Xenova/all-MiniLM-L6-v2");
  };

  // Function to generate embeddings
  const generateEmbedding = async () => {
    const result = await Pipeline.TextEmbedding.generate("Hello world");
    setEmbedding(result);
  };

  return (
    <View>
      <Button title="Load Model" onPress={loadModel} />
      <Button title="Generate Embedding" onPress={generateEmbedding} />
      <Text>Embedding Length: {embedding.length}</Text>
    </View>
  );
}
```

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
- [Babel Documentation](https://babeljs.io/)

These links provide additional information on how to configure and utilize the various components used by `react-native-transformers`.
