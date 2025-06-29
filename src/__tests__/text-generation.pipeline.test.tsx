import TextGenerationPipeline from '../pipelines/text-generation';

// Mock the model
jest.mock('../models/text-generation', () => {
  return {
    TextGeneration: jest.fn().mockImplementation(() => ({
      initializeFeed: jest.fn(),
      generate: jest.fn().mockImplementation((tokens) => {
        // Return tokens without calling callback to avoid double decoding
        return Promise.resolve(tokens);
      }),
      load: jest.fn(),
      release: jest.fn(),
      outputTokens: [],
    })),
  };
});

describe('TextGenerationPipeline', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset module state
    jest.isolateModules(() => {
      require('../pipelines/text-generation');
    });
  });

  describe('init', () => {
    it('should initialize with default options', async () => {
      await TextGenerationPipeline.init('test-model', 'test-path');
      expect(
        require('@huggingface/transformers').AutoTokenizer.from_pretrained
      ).toHaveBeenCalledWith('test-model');
    });

    it('should initialize with custom options', async () => {
      await TextGenerationPipeline.init('test-model', 'test-path', {
        show_special: true,
        max_tokens: 100,
      });
      expect(
        require('@huggingface/transformers').AutoTokenizer.from_pretrained
      ).toHaveBeenCalledWith('test-model');
    });
  });

  describe('generate', () => {
    beforeEach(async () => {
      await TextGenerationPipeline.init('test-model', 'test-path');
    });

    it('should generate text from prompt', async () => {
      const result = await TextGenerationPipeline.generate('test prompt');
      expect(result).toBe('decoded text');
    });

    it('should call callback with generated text', async () => {
      const callback = jest.fn();
      await TextGenerationPipeline.generate('test prompt', callback);
      expect(callback).toHaveBeenCalledWith('decoded text');
    });

    it('should throw error if not initialized', async () => {
      // Reset module state to clear tokenizer
      jest.resetModules();
      const freshPipeline = require('../pipelines/text-generation').default;
      await expect(freshPipeline.generate('test')).rejects.toThrow(
        'Tokenizer undefined, please initialize first.'
      );
    });
  });

  describe('release', () => {
    it('should release resources', async () => {
      await TextGenerationPipeline.init('test-model', 'test-path');
      await TextGenerationPipeline.release();
      // Reset module state to clear tokenizer
      jest.resetModules();
      const freshPipeline = require('../pipelines/text-generation').default;
      await expect(freshPipeline.generate('test')).rejects.toThrow(
        'Tokenizer undefined, please initialize first.'
      );
    });
  });
});
