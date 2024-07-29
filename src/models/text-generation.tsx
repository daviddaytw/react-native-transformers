import "text-encoding-polyfill";
import { Tensor } from "onnxruntime-react-native";
import { Base } from "./base";

/**
 * Class to handle a large language model on top of onnxruntime
 */
export class TextGeneration extends Base {
  public outputTokens: bigint[] = [];
  private needPositionIds = true;
  private stopGeneration = false;

  public initializeFeed() {
    super.initializeFeed();
    this.outputTokens = [];
  }

  /**
   * Generate tokens using greedy search
   *
   * @param tokens Initial tokens
   * @param callback Callback function to handle the generated tokens
   * @param options Generation options
   * @returns Array of generated tokens
   */
  public async generate(
    tokens: bigint[],
    callback: (tokens: bigint[]) => void,
    options: { maxTokens: number },
  ): Promise<bigint[]> {
    const maxTokens = options.maxTokens;
    const feed = this.feed;
    const initialTokens = BigInt64Array.from(tokens.map(BigInt));
    const inputIdsTensor = new Tensor("int64", initialTokens, [
      1,
      tokens.length,
    ]);
    feed.input_ids = inputIdsTensor;

    this.stopGeneration = false;
    this.outputTokens.push(...inputIdsTensor.data);

    let lastToken = 0n;
    let sequenceLength = this.outputTokens.length;
    const initialLength = feed.input_ids.size;

    // Prepare position IDs if needed
    if (this.needPositionIds) {
      feed.position_ids = new Tensor(
        "int64",
        BigInt64Array.from({ length: initialLength }, (_, i) =>
          BigInt(sequenceLength - initialLength + i),
        ),
        [1, initialLength],
      );
    }

    if (!this.sess) {
      throw new Error("Session is undefined");
    }

    // Generate tokens until the end of sequence token is found or max tokens limit is reached
    while (
      lastToken !== this.eos &&
      lastToken !== 32007n &&
      sequenceLength < maxTokens &&
      !this.stopGeneration
    ) {
      sequenceLength = this.outputTokens.length;

      feed.attention_mask = new Tensor(
        "int64",
        BigInt64Array.from({ length: sequenceLength }, () => 1n),
        [1, sequenceLength],
      );

      const outputs = await this.sess.run(feed);
      lastToken = BigInt(this.argmax(outputs.logits!));
      this.outputTokens.push(lastToken);

      if (callback) {
        callback(this.outputTokens);
      }

      this.updateKVCache(feed, outputs);
      feed.input_ids = new Tensor(
        "int64",
        BigInt64Array.from([lastToken]),
        [1, 1],
      );

      if (this.needPositionIds) {
        feed.position_ids = new Tensor(
          "int64",
          BigInt64Array.from([BigInt(sequenceLength)]),
          [1, 1],
        );
      }
    }
    return this.outputTokens;
  }
}
