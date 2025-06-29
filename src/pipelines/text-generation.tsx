import { env, AutoTokenizer } from '@huggingface/transformers';
import type { PreTrainedTokenizer } from '@huggingface/transformers';
import { TextGeneration as Model } from '../models/text-generation';
import type { LoadOptions } from '../models/base';

/** Initialization Options */
export interface InitOptions extends LoadOptions {
  /** Shows special tokens in the output. */
  show_special: boolean;
}

// Set up environment for transformers.js tokenizer
env.allowRemoteModels = true;
env.allowLocalModels = false;

// Declare tokenizer and model
let tokenizer: PreTrainedTokenizer;
const model: Model = new Model();

// Initialize options with default values
let _options: InitOptions = {
  show_special: false,
  max_tokens: 9999,
  fetch: async (url) => url,
  verbose: false,
  externalData: false,
  executionProviders: ['cpu'],
};

/**
 * Converts tokens to text using the initialized tokenizer.
 *
 * @param tokens - Array of tokens to convert.
 * @param startidx - Starting index for slicing the tokens array.
 * @returns The decoded text.
 */
function token_to_text(tokens: bigint[], startidx: number): string {
  if (!tokenizer) {
    throw new Error('Tokenizer undefined, please initialize first.');
  }

  return tokenizer.decode(tokens.slice(startidx) as unknown as number[], {
    skip_special_tokens: !_options.show_special,
  });
}

/**
 * Generates text based on the given prompt.
 *
 * @param prompt - The input prompt for text generation.
 * @param callback - Optional callback function to handle intermediate text.
 * @returns The generated text.
 */
async function generate(
  prompt: string,
  callback: (text: string) => void = () => {}
): Promise<string> {
  if (!tokenizer) {
    throw new Error('Tokenizer undefined, please initialize first.');
  }

  const { input_ids } = await tokenizer(prompt, {
    return_tensor: false,
    padding: true,
    truncation: true,
  });

  // Clear caches
  model.initializeFeed();
  let output_text = '';
  const record_output = (text: string) => {
    output_text += text;
    return text;
  };

  const output_index = model.outputTokens.length + input_ids.length;
  const output_tokens = await model.generate(
    input_ids.map(BigInt),
    (tokens) => {
      callback(record_output(token_to_text(tokens, output_index)));
    },
    { maxTokens: _options.max_tokens }
  );

  callback(record_output(token_to_text(output_tokens, output_index)));

  return output_text;
}

/**
 * Loads the model and tokenizer with the specified options.
 *
 * @param model_name - The name of the model to load.
 * @param onnx_path - The path to the ONNX model.
 * @param options - Optional initialization options.
 */
async function init(
  model_name: string,
  onnx_path: string,
  options?: Partial<InitOptions>
): Promise<void> {
  _options = { ..._options, ...options };
  tokenizer = await AutoTokenizer.from_pretrained(model_name);
  await model.load(model_name, onnx_path, _options);
}

/**
 * Releases the resources used by the model.
 */
async function release(): Promise<void> {
  await model.release();
}

// Export functions for external use
export default { init, generate, release };
