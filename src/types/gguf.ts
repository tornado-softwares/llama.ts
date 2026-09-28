import type { general_file_type } from '@/utils/constants';

export type architecture = 'llama' | 'mpt' | 'gptneox' | 'gptj' | 'gpt2' | 'bloom' | 'falcon' | 'mamba' | 'rwkv' | 'whisper';

export interface general {
    'general.architecture': architecture;
    'general.quantization_version'?: number;
    'general.alignment'?: number;
    'general.name'?: string;
    'general.author'?: string;
    'general.version'?: string;
    'general.organization'?: string;
    'general.basename'?: string;
    'general.finetune'?: string;
    'general.description'?: string;
    'general.quantized_by'?: string;
    'general.size_label'?: string;
    'general.license'?: string;
    'general.license.name'?: string;
    'general.license.link'?: string;
    'general.url'?: string;
    'general.doi'?: string;
    'general.uuid'?: string;
    'general.repo_url'?: string;
    'general.tags'?: string[];
    'general.languages'?: string[];
    'general.datasets'?: string[];
    'general.file_type'?: general_file_type;
    'general.source.url'?: string;
    'general.source.doi'?: string;
    'general.source.uuid'?: string;
    'general.source.repo_url'?: string;
    'general.base_model.count'?: number;
    [key: `general.base_model.${number}.${string}`]: unknown;
}

export interface llama extends general {
    'general.architecture': 'llama';
    'llama.context_length': number;
    'llama.embedding_length': number;
    'llama.block_count': number;
    'llama.feed_forward_length': number;
    'llama.rope.dimension_count': number;
    'llama.attention.head_count': number;
    'llama.attention.layer_norm_rms_epsilon': number;
    'llama.rope.scale'?: number;
    'llama.attention.head_count_kv'?: number;
    'llama.tensor_data_layout'?: string;
    'llama.expert_count'?: number;
    'llama.expert_used_count'?: number;
}

export interface mpt extends general {
    'general.architecture': 'mpt';
    'mpt.context_length': number;
    'mpt.embedding_length': number;
    'mpt.block_count': number;
    'mpt.attention.head_count': number;
    'mpt.attention.alibi_bias_max': number;
    'mpt.attention.clip_kqv': number;
    'mpt.attention.layer_norm_epsilon': number;
}

export interface gptneox extends general {
    'general.architecture': 'gptneox';
    'gptneox.context_length': number;
    'gptneox.embedding_length': number;
    'gptneox.block_count': number;
    'gptneox.use_parallel_residual': boolean;
    'gptneox.rope.dimension_count': number;
    'gptneox.attention.head_count': number;
    'gptneox.attention.layer_norm_epsilon': number;
    'gptneox.rope.scale'?: number;
}

export interface gptj extends general {
    'general.architecture': 'gptj';
    'gptj.context_length': number;
    'gptj.embedding_length': number;
    'gptj.block_count': number;
    'gptj.rope.dimension_count': number;
    'gptj.attention.head_count': number;
    'gptj.attention.layer_norm_epsilon': number;
    'gptj.rope.scale'?: number;
}

export interface gpt2 extends general {
    'general.architecture': 'gpt2';
    'gpt2.context_length': number;
    'gpt2.embedding_length': number;
    'gpt2.block_count': number;
    'gpt2.attention.head_count': number;
    'gpt2.attention.layer_norm_epsilon': number;
    'gpt2.feed_forward_length': number;
}

export interface bloom extends general {
    'general.architecture': 'bloom';
    'bloom.context_length': number;
    'bloom.embedding_length': number;
    'bloom.block_count': number;
    'bloom.feed_forward_length': number;
    'bloom.attention.head_count': number;
    'bloom.attention.layer_norm_epsilon': number;
}

export interface falcon extends general {
    'general.architecture': 'falcon';
    'falcon.context_length': number;
    'falcon.embedding_length': number;
    'falcon.block_count': number;
    'falcon.attention.head_count': number;
    'falcon.attention.head_count_kv': number;
    'falcon.attention.use_norm': boolean;
    'falcon.attention.layer_norm_epsilon': number;
    'falcon.tensor_data_layout'?: string;
}

export interface mamba extends general {
    'general.architecture': 'mamba';
    'mamba.context_length': number;
    'mamba.embedding_length': number;
    'mamba.block_count': number;
    'mamba.ssm.conv_kernel': number;
    'mamba.ssm.inner_size': number;
    'mamba.ssm.state_size': number;
    'mamba.ssm.time_step_rank': number;
    'mamba.attention.layer_norm_rms_epsilon': number;
}

export interface rwkv extends general {
    'general.architecture': 'rwkv';
    'rwkv.architecture_version': 4;
    'rwkv.context_length': number;
    'rwkv.block_count': number;
    'rwkv.embedding_length': number;
    'rwkv.feed_forward_length': number;
}

export interface whisper extends general {
    'general.architecture': 'whisper';
    'whisper.encoder.context_length': number;
    'whisper.encoder.embedding_length': number;
    'whisper.encoder.block_count': number;
    'whisper.encoder.mels_count': number;
    'whisper.encoder.attention.head_count': number;
    'whisper.decoder.context_length': number;
    'whisper.decoder.embedding_length': number;
    'whisper.decoder.block_count': number;
    'whisper.decoder.attention.head_count': number;
}

export type model_metadata = llama | mpt | gptneox | gptj | gpt2 | bloom | falcon | mamba | rwkv | whisper;

export type metadata = Record<string, unknown> & model_metadata;

export type metadata_for_architecture<A extends architecture> = Extract<model_metadata, { 'general.architecture': A }>;
