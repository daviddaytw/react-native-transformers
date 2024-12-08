import { TextEmbedding } from "../models/text-embedding";
import { InferenceSession } from "onnxruntime-react-native";

describe("TextEmbedding Model", () => {
  let model: TextEmbedding;

  beforeEach(() => {
    model = new TextEmbedding();
  });

  afterEach(async () => {
    await model.release();
  });

  it("should initialize properly", () => {
    expect(model).toBeInstanceOf(TextEmbedding);
  });

  it("should throw error when session is undefined", async () => {
    await expect(model.embed([1n, 2n, 3n])).rejects.toThrow(
      "Session is undefined",
    );
  });

  it("should throw error when no embedding output is found", async () => {
    // Mock session run to return empty outputs
    const mockRun = jest.fn().mockResolvedValue({});
    (model as any).sess = {
      run: mockRun,
      release: jest.fn().mockResolvedValue(undefined),
    } as Partial<InferenceSession>;

    await expect(model.embed([1n, 2n, 3n])).rejects.toThrow(
      "No embedding output found in model outputs",
    );
  });

  it("should properly calculate mean embeddings", async () => {
    // Mock session run to return sample embeddings
    const mockEmbeddings = new Float32Array([1, 2, 3, 4, 5, 6]); // 2 tokens, 3 dimensions
    const mockRun = jest.fn().mockResolvedValue({
      last_hidden_state: {
        data: mockEmbeddings,
        dims: [1, 2, 3], // [batch_size, sequence_length, hidden_size]
      },
    });
    (model as any).sess = {
      run: mockRun,
      release: jest.fn().mockResolvedValue(undefined),
    } as Partial<InferenceSession>;

    const result = await model.embed([1n, 2n]);

    // Expected mean values: [2.5, 3.5, 4.5]
    expect(Array.from(result)).toEqual([2.5, 3.5, 4.5]);
    expect(mockRun).toHaveBeenCalled();
  });
});
