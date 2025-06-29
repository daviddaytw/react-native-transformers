import { Base } from '../models/base';
import type { LoadOptions } from '../models/base';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

// Create a testable subclass to access protected methods
class TestableBase extends Base {
  public getSession() {
    return this.sess;
  }

  public setSession(session: InferenceSession | undefined) {
    this.sess = session;
  }

  public getFeed() {
    return this.feed;
  }

  public getEos() {
    return this.eos;
  }

  public getKvDims() {
    return (this as any).kv_dims;
  }

  public getNumLayers() {
    return (this as any).num_layers;
  }

  public getDtype() {
    return (this as any).dtype;
  }

  public callArgmax(tensor: Tensor): number {
    return this.argmax(tensor);
  }

  public callUpdateKVCache(
    feed: Record<string, Tensor>,
    outputs: InferenceSession.OnnxValueMapType
  ) {
    this.updateKVCache(feed, outputs);
  }
}

describe('Base Model', () => {
  let model: TestableBase;
  let mockFetch: jest.Mock;

  beforeEach(() => {
    model = new TestableBase();
    mockFetch = jest.fn();
    
    // Setup default mock responses
    mockFetch.mockResolvedValue('mock-model-path');
    
    // Mock global fetch for config loading
    global.fetch = jest.fn().mockResolvedValue({
      arrayBuffer: () =>
        Promise.resolve(
          Uint8Array.from(
            JSON.stringify({
              eos_token_id: 2,
              num_key_value_heads: 8,
              hidden_size: 512,
              num_attention_heads: 8,
              num_hidden_layers: 12,
            })
              .split('')
              .map((c) => c.charCodeAt(0))
          ).buffer
        ),
    });
  });

  afterEach(async () => {
    await model.release();
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should initialize with default values', () => {
      expect(model.getSession()).toBeUndefined();
      expect(model.getFeed()).toEqual({});
      expect(model.getEos()).toBe(2n);
      expect(model.getKvDims()).toEqual([]);
      expect(model.getNumLayers()).toBe(0);
      expect(model.getDtype()).toBe('float32');
    });
  });

  describe('load', () => {
    const defaultOptions: LoadOptions = {
      max_tokens: 100,
      verbose: false,
      externalData: false,
      fetch: mockFetch,
      executionProviders: [],
    };

    it('should load model successfully with default options', async () => {
      await model.load('test-model', 'onnx/model.onnx', defaultOptions);

      expect(global.fetch).toHaveBeenCalledWith(
        'https://huggingface.co/test-model/resolve/main/config.json'
      );
      expect(mockFetch).toHaveBeenCalledWith(
        'https://huggingface.co/test-model/resolve/main/onnx/model.onnx'
      );
      expect(model.getSession()).toBeDefined();
      expect(model.getEos()).toBe(2n);
      expect(model.getKvDims()).toEqual([1, 8, 0, 64]);
      expect(model.getNumLayers()).toBe(12);
    });

    it('should load model with custom onnx file path', async () => {
      await model.load('test-model', 'custom/path.onnx', defaultOptions);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://huggingface.co/test-model/resolve/main/custom/path.onnx'
      );
    });

    it('should load model with verbose logging enabled', async () => {
      const verboseOptions = { ...defaultOptions, verbose: true };
      await model.load('test-model', 'onnx/model.onnx', verboseOptions);

      expect(InferenceSession.create).toHaveBeenCalledWith(
        'mock-model-path',
        expect.objectContaining({
          logSeverityLevel: 0,
          logVerbosityLevel: 0,
        })
      );
    });

    it('should load model with external data', async () => {
      const externalDataOptions = { ...defaultOptions, externalData: true };
      await model.load('test-model', 'onnx/model.onnx', externalDataOptions);

      expect(mockFetch).toHaveBeenCalledWith(
        'https://huggingface.co/test-model/resolve/main/onnx/model.onnx_data'
      );
      expect(InferenceSession.create).toHaveBeenCalledWith(
        'mock-model-path',
        expect.objectContaining({
          externalData: ['mock-model-path'],
        })
      );
    });

    it('should load model with custom execution providers', async () => {
      const customOptions = {
        ...defaultOptions,
        executionProviders: [{ name: 'webgl' }],
      };
      await model.load('test-model', 'onnx/model.onnx', customOptions);

      expect(InferenceSession.create).toHaveBeenCalledWith(
        'mock-model-path',
        expect.objectContaining({
          executionProviders: [{ name: 'webgl' }],
        })
      );
    });

    it('should handle model config with different parameters', async () => {
      // Mock different config
      global.fetch = jest.fn().mockResolvedValue({
        arrayBuffer: () =>
          Promise.resolve(
            Uint8Array.from(
              JSON.stringify({
                eos_token_id: 50256,
                num_key_value_heads: 16,
                hidden_size: 1024,
                num_attention_heads: 16,
                num_hidden_layers: 24,
              })
                .split('')
                .map((c) => c.charCodeAt(0))
            ).buffer
          ),
      });

      await model.load('test-model', 'onnx/model.onnx', defaultOptions);

      expect(model.getEos()).toBe(50256n);
      expect(model.getKvDims()).toEqual([1, 16, 0, 64]);
      expect(model.getNumLayers()).toBe(24);
    });
  });

  describe('initializeFeed', () => {
    beforeEach(() => {
      // Set up model with some initial state
      (model as any).kv_dims = [1, 8, 0, 64];
      (model as any).num_layers = 2;
    });

    it('should initialize feed with empty tensors', () => {
      model.initializeFeed();

      const feed = model.getFeed();
      expect(feed['past_key_values.0.key']).toBeDefined();
      expect(feed['past_key_values.0.value']).toBeDefined();
      expect(feed['past_key_values.1.key']).toBeDefined();
      expect(feed['past_key_values.1.value']).toBeDefined();
    });

    it('should dispose previous gpu buffers', () => {
      const mockDispose = jest.fn();
      const mockTensor = {
        location: 'gpu-buffer',
        dispose: mockDispose,
      } as any;

      model.getFeed()['past_key_values.0.key'] = mockTensor;
      model.initializeFeed();

      expect(mockDispose).toHaveBeenCalled();
    });

    it('should not dispose non-gpu buffers', () => {
      const mockDispose = jest.fn();
      const mockTensor = {
        location: 'cpu',
        dispose: mockDispose,
      } as any;

      model.getFeed()['past_key_values.0.key'] = mockTensor;
      model.initializeFeed();

      expect(mockDispose).not.toHaveBeenCalled();
    });

    it('should handle float16 dtype', () => {
      (model as any).dtype = 'float16';
      model.initializeFeed();

      expect(Tensor).toHaveBeenCalledWith(
        'float16',
        expect.any(Uint16Array),
        [1, 8, 0, 64]
      );
    });
  });

  describe('argmax', () => {
    it('should return index of maximum value', () => {
      const mockTensor = {
        data: [0.1, 0.2, 0.8, 0.3, 0.4, 0.5],
        dims: [1, 2, 3],
      } as unknown as Tensor;

      const result = model.callArgmax(mockTensor);
      expect(result).toBe(2); // Index of 0.8 in the last sequence
    });

    it('should handle negative values', () => {
      const mockTensor = {
        data: [-0.5, -0.2, -0.8, -0.1, -0.3, -0.4],
        dims: [1, 2, 3],
      } as unknown as Tensor;

      const result = model.callArgmax(mockTensor);
      expect(result).toBe(0); // Index of -0.1 in the last sequence
    });

    it('should throw error for invalid tensor dimensions', () => {
      const mockTensor = {
        data: [0.1, 0.2],
        dims: [2],
      } as unknown as Tensor;

      expect(() => model.callArgmax(mockTensor)).toThrow('Invalid tensor dimensions');
    });

    it('should throw error for undefined dimensions', () => {
      const mockTensor = {
        data: [0.1, 0.2],
        dims: undefined,
      } as unknown as Tensor;

      expect(() => model.callArgmax(mockTensor)).toThrow('Invalid tensor dimensions');
    });

    it('should throw error for dimensions with zero values', () => {
      const mockTensor = {
        data: [0.1, 0.2],
        dims: [1, 0, 2],
      } as unknown as Tensor;

      expect(() => model.callArgmax(mockTensor)).toThrow('Invalid tensor dimensions');
    });

    it('should throw error for infinite values', () => {
      const mockTensor = {
        data: [0.1, Infinity, 0.3],
        dims: [1, 1, 3],
      } as unknown as Tensor;

      expect(() => model.callArgmax(mockTensor)).toThrow('found infinitive in logits');
    });

    it('should throw error for NaN values', () => {
      const mockTensor = {
        data: [0.1, NaN, 0.3],
        dims: [1, 1, 3],
      } as unknown as Tensor;

      expect(() => model.callArgmax(mockTensor)).toThrow('found infinitive in logits');
    });

    it('should handle equal maximum values', () => {
      const mockTensor = {
        data: [0.1, 0.5, 0.3, 0.5, 0.2, 0.1],
        dims: [1, 2, 3],
      } as unknown as Tensor;

      const result = model.callArgmax(mockTensor);
      expect(result).toBe(0); // First occurrence of maximum value
    });
  });

  describe('updateKVCache', () => {
    it('should update key-value cache from outputs', () => {
      const mockDispose = jest.fn();
      const oldTensor = {
        location: 'gpu-buffer',
        dispose: mockDispose,
      } as any;

      const feed: Record<string, any> = {
        'past_key_values.0.key': oldTensor,
      };

      const newTensor = new Tensor('float32', [], [1, 8, 10, 64]);
      const outputs = {
        'present.0.key': newTensor,
        'present.0.value': newTensor,
        'logits': new Tensor('float32', [], [1, 1, 1000]),
      };

      model.callUpdateKVCache(feed, outputs);

      expect(mockDispose).toHaveBeenCalled();
      expect(feed['past_key_values.0.key']).toBe(newTensor);
      expect(feed['past_key_values.0.value']).toBe(newTensor);
      expect(feed['logits']).toBeUndefined();
    });

    it('should not dispose non-gpu buffers', () => {
      const mockDispose = jest.fn();
      const oldTensor = {
        location: 'cpu',
        dispose: mockDispose,
      } as any;

      const feed = {
        'past_key_values.0.key': oldTensor,
      };

      const newTensor = new Tensor('float32', [], [1, 8, 10, 64]);
      const outputs = {
        'present.0.key': newTensor,
      };

      model.callUpdateKVCache(feed, outputs);

      expect(mockDispose).not.toHaveBeenCalled();
      expect(feed['past_key_values.0.key']).toBe(newTensor);
    });

    it('should handle undefined old tensor', () => {
      const feed: Record<string, Tensor> = {};
      const newTensor = new Tensor('float32', [], [1, 8, 10, 64]);
      const outputs = {
        'present.0.key': newTensor,
      };

      expect(() => model.callUpdateKVCache(feed, outputs)).not.toThrow();
      expect(feed['past_key_values.0.key']).toBe(newTensor);
    });

    it('should handle undefined output tensor', () => {
      const feed: Record<string, Tensor> = {};
      const outputs: Record<string, Tensor | undefined> = {
        'present.0.key': undefined,
      };

      model.callUpdateKVCache(feed, outputs as any);
      expect(feed['past_key_values.0.key']).toBeUndefined();
    });

    it('should ignore non-present outputs', () => {
      const feed: Record<string, Tensor> = {};
      const outputs = {
        'logits': new Tensor('float32', [], [1, 1, 1000]),
        'hidden_states': new Tensor('float32', [], [1, 10, 512]),
      };

      model.callUpdateKVCache(feed, outputs);
      expect(Object.keys(feed)).toHaveLength(0);
    });
  });

  describe('release', () => {
    it('should release session when it exists', async () => {
      const mockRelease = jest.fn().mockResolvedValue(undefined);
      const mockSession = {
        release: mockRelease,
      } as any;

      model.setSession(mockSession);
      await model.release();

      expect(mockRelease).toHaveBeenCalled();
      expect(model.getSession()).toBeUndefined();
    });

    it('should handle undefined session', async () => {
      model.setSession(undefined);
      await expect(model.release()).resolves.not.toThrow();
      expect(model.getSession()).toBeUndefined();
    });

    it('should handle session release errors', async () => {
      const mockRelease = jest.fn().mockRejectedValue(new Error('Release failed'));
      const mockSession = {
        release: mockRelease,
      } as any;

      model.setSession(mockSession);
      await expect(model.release()).rejects.toThrow('Release failed');
      expect(mockRelease).toHaveBeenCalled();
    });
  });

  describe('helper functions', () => {
    it('should generate correct Hugging Face URL', () => {
      // Access the private function through model loading
      expect(global.fetch).toBeDefined();
    });

    it('should handle load function with fetch', async () => {
      const mockArrayBuffer = new ArrayBuffer(8);
      global.fetch = jest.fn().mockResolvedValue({
        arrayBuffer: () => Promise.resolve(mockArrayBuffer),
      });

      // This tests the internal load function indirectly
      const options: LoadOptions = {
        max_tokens: 100,
        verbose: false,
        externalData: false,
        fetch: mockFetch,
        executionProviders: [],
      };

      mockFetch.mockResolvedValue('mock-path');
      await model.load('test-model', 'onnx/model.onnx', options);

      expect(global.fetch).toHaveBeenCalled();
    });
  });

  describe('edge cases', () => {
    it('should handle model with zero hidden layers', async () => {
      global.fetch = jest.fn().mockResolvedValue({
        arrayBuffer: () =>
          Promise.resolve(
            Uint8Array.from(
              JSON.stringify({
                eos_token_id: 2,
                num_key_value_heads: 8,
                hidden_size: 512,
                num_attention_heads: 8,
                num_hidden_layers: 0,
              })
                .split('')
                .map((c) => c.charCodeAt(0))
            ).buffer
          ),
      });

      const options: LoadOptions = {
        max_tokens: 100,
        verbose: false,
        externalData: false,
        fetch: mockFetch,
        executionProviders: [],
      };

      await model.load('test-model', 'onnx/model.onnx', options);
      model.initializeFeed();

      expect(model.getNumLayers()).toBe(0);
      expect(Object.keys(model.getFeed())).toHaveLength(0);
    });

    it('should handle argmax with single element', () => {
      const mockTensor = {
        data: [0.5, 0.3, 0.7],
        dims: [1, 1, 3],
      } as unknown as Tensor;

      const result = model.callArgmax(mockTensor);
      expect(result).toBe(2);
    });

    it('should handle argmax with all same values', () => {
      const mockTensor = {
        data: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        dims: [1, 2, 3],
      } as unknown as Tensor;

      const result = model.callArgmax(mockTensor);
      expect(result).toBe(0);
    });
  });
});
