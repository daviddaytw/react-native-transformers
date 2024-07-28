import "text-encoding-polyfill";
import { env, InferenceSession, Tensor } from "onnxruntime-react-native";

async function load(uri: string) {
  // @ts-ignore
  return await fetch(uri).then((response) => response.arrayBuffer());
}

function getHuggingfaceUrl(model: string, filepath: string) {
  return "https://huggingface.co/" + model + "/resolve/main" + filepath;
}

export interface LoadOptions {
  max_tokens: number;
  verbose: boolean;
  profiler: boolean;
  externalData: boolean;
  fetch: (url: string) => Promise<string>;
  executionProviders: InferenceSession.ExecutionProviderConfig[];
}

/**
 * Class to handle a large language model on top of onnxruntime
 */
export class LLM {
  sess?: InferenceSession = undefined;
  profiler = false;
  feed: Record<string, Tensor> = {};
  output_tokens: bigint[] = [];
  eos = 2n;
  need_position_ids = true;
  stop = false;
  kv_dims: number[] = [];
  num_layers: number = 0;
  dtype: "float16" | "float32" = "float32";
  max_tokens = 9999;

  constructor() {}

  async load(
    model: string,
    onnx_file: string = "/onnx/model.onnx",
    options: LoadOptions,
  ) {
    const verbose = options.verbose;
    this.profiler = options.profiler;
    const { fetch } = options;

    const json_bytes = await load(
      await fetch(getHuggingfaceUrl(model, "/config.json")),
    );
    // @ts-ignore
    let textDecoder = new TextDecoder();
    const model_config = JSON.parse(textDecoder.decode(json_bytes));
    const model_path = await fetch(getHuggingfaceUrl(model, onnx_file));

    const opt: InferenceSession.SessionOptions = {
      executionProviders: options.executionProviders,
      graphOptimizationLevel: "all",
    };

    if (options.externalData) {
      opt.externalData = [
        await fetch(getHuggingfaceUrl(model, onnx_file + "_data")),
      ];
    }
    if (verbose) {
      opt.logSeverityLevel = 0;
      opt.logVerbosityLevel = 0;
      env.logLevel = "verbose";
    }

    if (this.profiler) {
      opt.enableProfiling = true;
    }

    this.sess = await InferenceSession.create(model_path, opt);
    this.eos = model_config.eos_token_id;
    this.kv_dims = [
      1,
      model_config.num_key_value_heads,
      0,
      model_config.hidden_size / model_config.num_attention_heads,
    ];
    this.num_layers = model_config.num_hidden_layers;
    this.initilize_feed();
  }

  initilize_feed() {
    const feed = this.feed;

    // dispose of previous gpu buffers
    for (const name in feed) {
      const t = feed[name];
      if (t !== undefined && t.location === "gpu-buffer") {
        t.dispose();
      }
    }
    this.feed = {};
    // key value cache is zero copy, just pass gpu buffer as referece
    const empty = this.dtype === "float16" ? new Uint16Array() : [];
    for (let i = 0; i < this.num_layers; ++i) {
      this.feed[`past_key_values.${i}.key`] = new Tensor(
        this.dtype,
        empty,
        this.kv_dims,
      );
      this.feed[`past_key_values.${i}.value`] = new Tensor(
        this.dtype,
        empty,
        this.kv_dims,
      );
    }
    this.output_tokens = [];
  }

  /**
   * poor mens argmax
   *
   * @param t
   * @returns
   */
  argmax(t: Tensor) {
    const arr = t.data;
    const start = t.dims[2] * (t.dims[1] - 1);
    let max = arr[start];
    let maxidx = 0;

    for (let i = 0; i < t.dims[2]; i++) {
      const val = arr[i + start];
      if (!isFinite(val as number)) {
        throw new Error("found infinitive in logits");
      }
      if (val > max) {
        max = arr[i + start];
        maxidx = i;
      }
    }
    return maxidx;
  }

  //
  // update key value cache
  //
  update_kv_cache(
    feed: Record<string, Tensor>,
    outputs: InferenceSession.OnnxValueMapType,
  ) {
    for (const name in outputs) {
      if (name.startsWith("present")) {
        let newName = name.replace("present", "past_key_values");
        // dispose previous gpu buffers
        const t = feed[newName];
        if (t !== undefined && t.location === "gpu-buffer") {
          t.dispose();
        }
        feed[newName] = outputs[name];
      }
    }
  }

  async release() {
    if (this.sess !== undefined) {
      await this.sess.release();
    }
    this.sess = undefined;
  }

  //
  // prefill prompt and generate tokens, greedy search only
  //
  async generate(
    tokens: bigint[],
    callback: (tokens: bigint[]) => void,
    options: { max_tokens: number },
  ) {
    const max_tokens = options.max_tokens || 256;
    const feed = this.feed;
    const input_ids = new Tensor(
      "int64",
      BigInt64Array.from(tokens.map(BigInt)),
      [1, tokens.length],
    );
    feed.input_ids = input_ids;
    this.stop = false;

    this.output_tokens.push(...input_ids.data);

    let last_token = 0n;
    let seqlen = this.output_tokens.length;
    const input_len = input_ids.size;

    if (this.need_position_ids) {
      feed.position_ids = new Tensor(
        "int64",
        BigInt64Array.from({ length: input_len }, (_, i) =>
          BigInt(seqlen - input_len + i),
        ),
        [1, input_len],
      );
    }

    if (this.sess === undefined) {
      throw new Error("Session is undefined");
    }

    while (
      last_token !== this.eos &&
      last_token !== 32007n &&
      seqlen < max_tokens &&
      !this.stop
    ) {
      seqlen = this.output_tokens.length;
      feed.attention_mask = new Tensor(
        "int64",
        BigInt64Array.from({ length: seqlen }, () => 1n),
        [1, seqlen],
      );
      const outputs = await this.sess.run(feed);
      last_token = BigInt(this.argmax(outputs.logits!));
      this.output_tokens.push(last_token);
      if (callback && !this.profiler) {
        callback(this.output_tokens);
      }
      this.update_kv_cache(feed, outputs);
      feed.input_ids = new Tensor(
        "int64",
        BigInt64Array.from([last_token]),
        [1, 1],
      );
      if (this.need_position_ids) {
        feed.position_ids = new Tensor(
          "int64",
          BigInt64Array.from([BigInt(seqlen)]),
          [1, 1],
        );
      }
    }
    if (this.profiler) {
      this.sess.endProfiling();
    }
    return this.output_tokens;
  }
}
