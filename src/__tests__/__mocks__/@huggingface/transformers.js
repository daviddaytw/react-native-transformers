module.exports = {
  env: { allowRemoteModels: true, allowLocalModels: false },
  AutoTokenizer: {
    from_pretrained: jest.fn().mockResolvedValue(
      Object.assign(
        jest.fn((_text, _options) => ({ input_ids: [1, 2, 3, 4] })),
        {
          decode: jest.fn((_tokens, _options) => 'decoded text'),
          encode: jest.fn((_text, _options) => ({ input_ids: [1, 2, 3, 4] })),
          call: jest.fn((_text, _options) => ({ input_ids: [1, 2, 3, 4] })),
        }
      )
    ),
  },
};
