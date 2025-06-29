import TextEmbeddingPipeline from '../pipelines/text-embedding';

// Mock the TextEmbedding model
jest.mock('../models/text-embedding', () => {
  return {
    TextEmbedding: jest.fn().mockImplementation(() => ({
      load: jest.fn().mockResolvedValue(undefined),
      embed: jest.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3])),
      release: jest.fn().mockResolvedValue(undefined),
    })),
  };
});

describe('TextEmbedding Pipeline', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(async () => {
    await TextEmbeddingPipeline.release();
  });

  it('should throw error when not initialized', async () => {
    await expect(TextEmbeddingPipeline.embed('test text')).rejects.toThrow(
      'Tokenizer undefined, please initialize first'
    );
  });

  it('should initialize properly', async () => {
    await expect(
      TextEmbeddingPipeline.init('test-model', 'model.onnx')
    ).resolves.not.toThrow();
  });

  it('should generate embeddings', async () => {
    await TextEmbeddingPipeline.init('test-model', 'model.onnx');
    const embeddings = await TextEmbeddingPipeline.embed('test text');
    expect(embeddings).toBeInstanceOf(Float32Array);
    expect(embeddings.length).toBe(3);
  });
});
