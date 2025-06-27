// Mock text-encoding-polyfill
jest.mock('text-encoding-polyfill', () => ({}));

// Mock fetch for config loading
global.fetch = jest.fn(() =>
  Promise.resolve({
    arrayBuffer: () =>
      Promise.resolve(
        Uint8Array.from(
          JSON.stringify({
            eos_token_id: 2,
            num_key_value_heads: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_hidden_layers: 32,
          })
            .split('')
            .map((c) => c.charCodeAt(0))
        ).buffer
      ),
  })
);

// Mock InferenceSession
jest.mock('onnxruntime-react-native', () => ({
  InferenceSession: {
    create: jest.fn().mockResolvedValue({
      run: jest.fn().mockResolvedValue({
        logits: {
          data: new Float32Array([0.1, 0.2, 0.3, 0.4]),
          dims: [1, 1, 4],
          type: 'float32',
        },
      }),
      release: jest.fn(),
    }),
  },
  env: { logLevel: 'error' },
  Tensor: jest.fn().mockImplementation((type, data, dims) => ({
    type,
    data,
    dims,
    size: data.length,
    dispose: jest.fn(),
  })),
}));

// Mock transformers
jest.mock('@huggingface/transformers', () => ({
  env: { allowRemoteModels: true, allowLocalModels: false },
  AutoTokenizer: {
    from_pretrained: jest.fn().mockResolvedValue({
      decode: jest.fn((_tokens, _options) => 'decoded text'),
    }),
  },
}));
