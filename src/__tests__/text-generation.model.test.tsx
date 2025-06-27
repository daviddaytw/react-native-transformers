import { TextGeneration } from '../models/text-generation';
import { Tensor } from 'onnxruntime-react-native';

// Mock onnxruntime-react-native
jest.mock('onnxruntime-react-native', () => ({
  Tensor: jest.fn().mockImplementation((type, data, dims) => ({
    type,
    data,
    dims,
    size: data.length,
  })),
}));

// Create a test-specific subclass to access protected properties
class TestableTextGeneration extends TextGeneration {
  public getSession() {
    return this.sess;
  }

  public setSession(session: any) {
    this.sess = session;
  }

  public getFeed() {
    return this.feed;
  }

  public getEos() {
    return this.eos;
  }
}

describe('TextGeneration Model', () => {
  let model: TestableTextGeneration;
  let mockRunCount: number;

  beforeEach(() => {
    mockRunCount = 0;
    model = new TestableTextGeneration();
  });

  describe('initializeFeed', () => {
    it('should reset output tokens', () => {
      model.outputTokens = [1n, 2n, 3n];
      model.initializeFeed();
      expect(model.outputTokens).toEqual([]);
    });
  });

  describe('generate', () => {
    const mockCallback = jest.fn();
    const mockTokens = [1n, 2n]; // Initial tokens

    beforeEach(() => {
      mockCallback.mockClear();
      mockRunCount = 0;
    });

    it('should generate tokens until EOS token is found', async () => {
      model.setSession({
        run: jest.fn().mockImplementation(() => {
          mockRunCount++;
          return Promise.resolve({
            logits: {
              data: new Float32Array([0.1, 0.2, 0.3, 2.0]), // highest value at index 3
              dims: [1, 1, 4],
              type: 'float32',
            },
          });
        }),
      });

      const result = await model.generate(mockTokens, mockCallback, {
        maxTokens: 10,
      });
      expect(result.length).toBeGreaterThan(0);
      expect(mockCallback).toHaveBeenCalled();
    });

    it('should respect maxTokens limit', async () => {
      const maxTokens = 5;
      model.setSession({
        run: jest.fn().mockImplementation(() => {
          mockRunCount++;
          return Promise.resolve({
            logits: {
              data: new Float32Array([0.1, 0.2, 0.3, 0.1]), // will generate token 2 (index with highest value)
              dims: [1, 1, 4],
              type: 'float32',
            },
          });
        }),
      });

      const result = await model.generate(mockTokens, mockCallback, {
        maxTokens,
      });
      // Initial tokens (2) + generated tokens should not exceed maxTokens (5)
      expect(result.length).toBeLessThanOrEqual(maxTokens);
      expect(mockRunCount).toBeLessThanOrEqual(maxTokens - mockTokens.length);
    });

    it('should throw error if session is undefined', async () => {
      model.setSession(undefined);
      await expect(
        model.generate(mockTokens, mockCallback, { maxTokens: 10 })
      ).rejects.toThrow('Session is undefined');
    });

    it('should create correct tensors for input', async () => {
      model.setSession({
        run: jest.fn().mockResolvedValue({
          logits: {
            data: new Float32Array([0.1, 0.2, 0.3, 0.4]),
            dims: [1, 1, 4],
            type: 'float32',
          },
        }),
      });

      await model.generate(mockTokens, mockCallback, { maxTokens: 10 });
      expect(Tensor).toHaveBeenCalledWith('int64', expect.any(BigInt64Array), [
        1,
        mockTokens.length,
      ]);
    });

    it('should handle generation with attention mask', async () => {
      model.setSession({
        run: jest.fn().mockResolvedValue({
          logits: {
            data: new Float32Array([0.1, 0.2, 0.3, 0.4]),
            dims: [1, 1, 4],
            type: 'float32',
          },
        }),
      });

      const result = await model.generate(mockTokens, mockCallback, {
        maxTokens: 10,
      });
      const feed = model.getFeed();
      expect(feed.attention_mask).toBeDefined();
      expect(result).toBeDefined();
    });
  });

  describe('release', () => {
    it('should release session resources', async () => {
      const mockSession = {
        release: jest.fn().mockResolvedValue(undefined),
      };
      model.setSession(mockSession);

      await model.release();
      expect(mockSession.release).toHaveBeenCalled();
      expect(model.getSession()).toBeUndefined();
    });
  });
});
