# react-native-transformers

![NPM Version](https://img.shields.io/npm/v/react-native-transformers)
[![codecov](https://codecov.io/github/daviddaytw/react-native-transformers/graph/badge.svg?token=G3D0Y33SI4)](https://codecov.io/github/daviddaytw/react-native-transformers)
[![TypeDoc](https://github.com/daviddaytw/react-native-transformers/actions/workflows/docs.yml/badge.svg)](https://daviddaytw.github.io/react-native-transformers)

`react-native-transformers` is a React Native library for incorporating transformer models into your mobile applications. It supports both iOS and Android platforms, allowing you to leverage advanced AI models directly on your device.

## Features

- On-device transformer model support
- Compatible with iOS and Android
- Simple API for model loading and text generation

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

Here is an example of how to use `react-native-transformers` in an Expo application:

```javascript
import React from "react";
import { View, Text, Button, TextInput } from "react-native";
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

For detailed usage please reference ['API documentation'](https://daviddaytw.github.io/react-native-transformers/)

## Contributing

Contributions are welcome! See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This library relies on the ONNX Runtime for running the models efficiently on mobile devices.

## External Links

- [Expo Plugins](https://docs.expo.dev/guides/config-plugins/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Babel](https://babeljs.io/)
- [Hugging Face](https://huggingface.co/)

These links provide additional information on how to configure and utilize the various components used by `react-native-transformers`.
