module.exports = {
  preset: "react-native",
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json", "node"],
  transformIgnorePatterns: [
    "node_modules/(?!(" +
      "react-native|" +
      "@react-native|" +
      "@xenova/transformers|" +
      "text-encoding-polyfill" +
      ")/)",
  ],
  setupFiles: [
    "./node_modules/react-native/jest/setup.js",
    "./src/__tests__/setup.js",
  ],
  testRegex: "(/__tests__/.*(?<!setup)\\.(test|spec))\\.[jt]sx?$",
  testEnvironment: "node",
  transform: {
    "^.+\\.(js|jsx|ts|tsx)$": [
      "babel-jest",
      { configFile: "./babel.config.js" },
    ],
  },
  globals: {
    "ts-jest": {
      babelConfig: true,
      tsconfig: "tsconfig.json",
    },
  },
  collectCoverage: true,
  coverageDirectory: "coverage",
  coverageReporters: ["text", "lcov"],
  collectCoverageFrom: [
    "src/**/*.{js,jsx,ts,tsx}",
    "!src/**/*.d.ts",
    "!src/__tests__/**",
  ],
};
