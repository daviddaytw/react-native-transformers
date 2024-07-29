import { TextGeneration as TextGenerationModel } from "./models/text-generation";
import TextGenerationPipeline from "./pipelines/text-generation";

export const Pipeline = {
  TextGeneration: TextGenerationPipeline,
};

export const Model = {
  TextGeneration: TextGenerationModel,
};

export default {
  Pipeline,
  Model,
};
