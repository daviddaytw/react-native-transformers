declare module '@huggingface/transformers' {
  export interface PreTrainedTokenizer {
    (
      text: string,
      options?: {
        return_tensor?: boolean;
        padding?: boolean;
        truncation?: boolean;
        max_length?: number;
      }
    ): Promise<{ input_ids: number[] }>;
    decode(
      tokens: number[],
      options?: { skip_special_tokens?: boolean }
    ): string;
  }

  export class AutoTokenizer {
    static from_pretrained(model_name: string): Promise<PreTrainedTokenizer>;
  }

  export const env: {
    allowRemoteModels: boolean;
    allowLocalModels: boolean;
    logLevel?: string;
  };
}
