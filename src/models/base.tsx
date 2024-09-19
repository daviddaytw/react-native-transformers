import "text-encoding-polyfill";
import { env, InferenceSession, Tensor } from "onnxruntime-react-native";

async function load(uri: string): Promise<ArrayBuffer> {
  // @ts-ignore
  return await fetch(uri).then((response) => response.arrayBuffer());
}

function getHuggingfaceUrl(model: string, filepath: string): string {
  return `https://huggingface.co/${model}/resolve/main/${filepath}`;
}

/** Load Options */
export interface LoadOptions {
  /** The maximum number of tokens for text generation.  */
  max_tokens: number;
  /** Enables verbose logging. */
  verbose: boolean;
  /** Indicates if external data is used. */
  externalData: boolean;
  /** Function to fetch external data. */
  fetch: (url: string) => Promise<string>;
  /** List of execution providers for ONNX runtime. */
  executionProviders: InferenceSession.ExecutionProviderConfig[];
}

export class Base {
  protected sess?: InferenceSession;
  protected feed: Record<string, Tensor> = {};
  protected eos = 2n;
  private kv_dims: number[] = [];
  private num_layers = 0;
  private dtype: "float16" | "float32" = "float32";

  constructor() {}

  async load(
    model: string,
    onnx_file: string = "onnx/model.onnx",
    options: LoadOptions,
  ) {
    const verbose = options.verbose;
    const fetch = options.fetch;

    const json_bytes = await load(
      await fetch(getHuggingfaceUrl(model, "config.json")),
    );
    // @ts-ignore
    const textDecoder = new TextDecoder();
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

    this.sess = await InferenceSession.create(model_path, opt);
    this.eos = model_config.eos_token_id;
    this.kv_dims = [
      1,
      model_config.num_key_value_heads,
      0,
      model_config.hidden_size / model_config.num_attention_heads,
    ];
    this.num_layers = model_config.num_hidden_layers;
    this.initializeFeed();
  }

  public initializeFeed() {
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
    for (let i = 0; i < this.num_layers; i++) {
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
  }

  protected argmax(t: Tensor): number {
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
        max = val;
        maxidx = i;
      }
    }
    return maxidx;
  }

  protected updateKVCache(
    feed: Record<string, Tensor>,
    outputs: InferenceSession.OnnxValueMapType,
  ) {
    for (const name in outputs) {
      if (name.startsWith("present")) {
        const newName = name.replace("present", "past_key_values");
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
    if (this.sess) {
      await this.sess.release();
      this.sess = undefined;
    }
  }
}
