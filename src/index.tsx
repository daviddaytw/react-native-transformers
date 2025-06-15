import { TextGeneration } from './models/text-generation';
import TextGenerationPipeline from './pipelines/text-generation';

export const Pipeline = { TextGeneration: TextGenerationPipeline };

export const Model = { TextGeneration };

export default { Pipeline, Model };

export type * from './models/base';
export type * from './models/text-generation';
export type * from './pipelines/text-generation';
