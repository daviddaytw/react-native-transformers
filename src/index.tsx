import { TextGeneration } from "./models/text-generation";
import { TextEmbedding } from "./models/text-embedding";
import TextGenerationPipeline from "./pipelines/text-generation";
import TextEmbeddingPipeline from "./pipelines/text-embedding";

export const Pipeline = {
  TextGeneration: TextGenerationPipeline,
  TextEmbedding: TextEmbeddingPipeline,
};

export const Model = {
  TextGeneration,
  TextEmbedding,
};

export default {
  Pipeline,
  Model,
};

export type * from "./models/base";
export type * from "./models/text-generation";
export type * from "./models/text-embedding";
export type * from "./pipelines/text-generation";
export type * from "./pipelines/text-embedding";
