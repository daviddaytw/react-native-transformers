// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');
const path = require('path')

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

const packagePath = path.resolve(__dirname, '../')
console.log('Package path')
config.resolver.extraNodeModules['react-native-transformers'] = packagePath
config.watchFolders = [...config.watchFolders, packagePath]

module.exports = config;
