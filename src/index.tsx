import { env, AutoTokenizer, PreTrainedTokenizer } from "@xenova/transformers";
import { LLM, type LoadOptions } from "./llm";

interface InitOptions extends LoadOptions {
  show_special: boolean;
}

// setup for transformers.js tokenizer
env.allowRemoteModels = true;
env.allowLocalModels = false;

let tokenizer: PreTrainedTokenizer;
const llm: LLM = new LLM();

let _options: InitOptions = {
  show_special: false,
  max_tokens: 9999,
  fetch: async (url) => url,
  verbose: false,
  profiler: false,
  externalData: false,
  executionProviders: ["cpu"],
};

function token_to_text(tokens: bigint[], startidx: number) {
  if (tokenizer === undefined) {
    throw new Error("Tokenizer undefined, please initialize first.");
  }

  const txt = tokenizer.decode(tokens.slice(startidx) as unknown as number[], {
    skip_special_tokens: _options.show_special,
  });
  return txt;
}

async function text_generation(
  prompt: string,
  callback: (text: string) => void = () => {},
) {
  if (tokenizer === undefined) {
    throw new Error("Tokenizer undefined, please initialize first.");
  }

  const { input_ids } = await tokenizer(prompt, {
    return_tensor: false,
    padding: true,
    truncation: true,
  });

  // clear caches
  llm.initilize_feed();
  let output_text = "";
  const record_output = (text: string) => {
    output_text += text;
    return text;
  };

  const output_index = llm.output_tokens.length + input_ids.length;
  const output_tokens = await llm.generate(
    input_ids,
    (tokens) => {
      callback(record_output(token_to_text(tokens, output_index)));
    },
    { max_tokens: _options.max_tokens },
  );

  callback(record_output(token_to_text(output_tokens, output_index)));

  return output_text;
}

/**
 * Load the model and tokenizer
 *
 * @param model
 * @param onnx_path
 * @param options
 */
async function init(
  model: string,
  onnx_path: string,
  options?: Partial<InitOptions>,
) {
  _options = Object.assign(_options, options);
  tokenizer = await AutoTokenizer.from_pretrained(model);
  await llm.load(model, onnx_path, _options);
}

async function release() {
  await llm.release();
}

export default {
  init,
  text_generation,
  release,
};
