import { decode, encode } from 'gpt-tokenizer/model/gpt-2';
import { kv_cache } from '@/lib/kv-cache';
import { model } from '@/lib/model';
import type { tensor } from '@/lib/tensor';
import type { transformer_block } from '@/lib/transformer-block';
import type { gpt2 as _gpt2 } from '@/types/gguf';

export class gpt2 extends model {
    private context_length;
    private embedding_length;
    private feed_forward_length;
    private block_count;
    private attention_head_count;
    private attention_layer_norm_epsilon;

    private token_embedding_tensor!: tensor;
    private position_embedding_tensor!: tensor;
    private output_norm_weight_tensor!: tensor;
    private output_norm_bias_tensor!: tensor;
    private lm_head_tensor!: tensor;

    private transformer_blocks: transformer_block[] = [];
    private kv_caches: kv_cache[] = [];

    constructor(metadata: _gpt2, tensors: tensor[], transformer_blocks: transformer_block[]) {
        super();
        this.context_length = metadata['gpt2.context_length'];
        this.embedding_length = metadata['gpt2.embedding_length'];
        this.feed_forward_length = metadata['gpt2.feed_forward_length'];
        this.block_count = metadata?.['gpt2.block_count'] ?? transformer_blocks.length;
        this.attention_head_count = metadata['gpt2.attention.head_count'];
        this.attention_layer_norm_epsilon = metadata['gpt2.attention.layer_norm_epsilon'] ?? 1e5;
        this.transformer_blocks = transformer_blocks;

        console.log(metadata);

        for (const tensor of tensors) {
            if (tensor.name === 'token_embd.weight') this.token_embedding_tensor = tensor;
            if (tensor.name === 'position_embd.weight') this.position_embedding_tensor = tensor;
            if (tensor.name === 'output_norm.weight') this.output_norm_weight_tensor = tensor;
            if (tensor.name === 'output_norm.bias') this.output_norm_bias_tensor = tensor;
            if (tensor.name === 'output.weight' || tensor.name === 'lm_head.weight') {
                this.lm_head_tensor = tensor;
            }
        }

        /*
         * https://discuss.huggingface.co/t/why-is-the-lm-head-layer-in-gpt2lmheadmodel-not-a-parameter/639
         * Need to transpose lm_head tensor since it is weight tied with token embedding tensor ?
         * https://github.com/huggingface/transformers/blob/19e5ed736611227b004c6f55679ce3536db3c28d/src/transformers/models/gpt2/modeling_gpt2.py#L943
         * _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]
         */
        if (!this.lm_head_tensor) this.lm_head_tensor = this.token_embedding_tensor;

        const hidden_size = this.embedding_length;
        for (let i = 0; i < this.block_count; i++) {
            this.kv_caches.push(
                new kv_cache({
                    size: this.context_length * hidden_size,
                    embedding_length: this.embedding_length,
                })
            );
        }
    }

    private load_tensors() {
        if (this.token_embedding_tensor.status === 'cold') this.token_embedding_tensor.load();
        if (this.position_embedding_tensor.status === 'cold') this.position_embedding_tensor.load();
        if (this.output_norm_weight_tensor.status === 'cold') this.output_norm_weight_tensor.load();
        if (this.output_norm_bias_tensor.status === 'cold') this.output_norm_bias_tensor.load();
        if (this.lm_head_tensor.status === 'cold') this.lm_head_tensor.load();
    }

    encode(input: string): number[] {
        return encode(input);
    }

    decode(input: number[]): string {
        return decode(input);
    }

    forward(tokens: number[], start_position: number) {
        console.time('New token');
        this.load_tensors();
        let hidden = this.device.embed({
            start_position: start_position,
            token_embedding: this.token_embedding_tensor,
            position_embedding: this.position_embedding_tensor,
            tokens: tokens,
        });
        for (let b = 0; b < this.block_count; b++) {
            const block = this.transformer_blocks[b];
            const cache = this.kv_caches[b];
            block.load();

            //        attn_norm_weight_tensor.shape : [ 768 ] Vector            attn_norm_bias_tensor.shape : [ 768 ] Vector
            //        attn_norm_weight_tensor.data  : [ 768-f32 ]               attn_norm_bias_tensor.data  : [ 768-f32 ]
            const norm1 = this.device.layer_norm({
                input: hidden,
                weight: block.get('attn_norm.weight'),
                bias: block.get('attn_norm.bias'),
                epsilon: this.attention_layer_norm_epsilon,
            });

            //
            //         We do not cache Q because old queries                   ---- CACHE ----
            //         are useless for the next prediction.                    |             |
            //                                                Q (query)      K (key)     V (value)
            //                                                   ↑             ↑             ↑
            //        attn_qkv_weight_tensor.data : 768 x [ [ 768-f32 ] + [ 768-f32 ] + [ 768-f32 ] ] 2D
            //        attn_qkv_bias_tensor.data   :  1  x [ [ 768-f32 ] + [ 768-f32 ] + [ 768-f32 ] ] Vector
            //
            const qkv = this.device.linear({
                input: norm1,
                weight: block.get('attn_qkv.weight'),
                bias: block.get('attn_qkv.bias'),
            });
            const [query, keys, values] = this.device.split_columns(qkv, 3);
            cache.set(keys.data, values.data, tokens.length);
            const attn = this.device.causal_attention({
                query: query.data,
                keys: cache.keys,
                values: cache.values,
                n_tokens: tokens.length,
                total_len: cache.current_length,
                start_position: start_position,
                num_heads: this.attention_head_count,
                hidden_size: this.embedding_length,
            });
            const proj = this.device.linear({
                input: attn,
                weight: block.get('attn_output.weight'),
                bias: block.get('attn_output.bias'),
            });
            const res1 = this.device.add(proj, hidden);
            const norm2 = this.device.layer_norm({
                input: res1,
                weight: block.get('ffn_norm.weight'),
                bias: block.get('ffn_norm.bias'),
                epsilon: this.attention_layer_norm_epsilon,
            });
            const up = this.device.gelu(
                this.device.linear({
                    input: norm2,
                    weight: block.get('ffn_up.weight'),
                    bias: block.get('ffn_up.bias'),
                })
            );
            const down = this.device.linear({
                input: up,
                weight: block.get('ffn_down.weight'),
                bias: block.get('ffn_down.bias'),
            });
            hidden = this.device.add(down, res1);
        }
        const last = this.device.take_token(hidden, tokens.length - 1);
        const normed = this.device.layer_norm({
            input: last,
            weight: this.output_norm_weight_tensor,
            bias: this.output_norm_bias_tensor,
            epsilon: this.attention_layer_norm_epsilon,
        });
        const logits = this.device.linear({
            input: normed,
            weight: this.lm_head_tensor,
        });
        console.timeEnd('New token');
        return logits; //this.device.argmax(logits.data);
    }

    public inference({ prompt, max_tokens, temperature }: { prompt: string; max_tokens: number; temperature: number }): string {
        this.kv_caches.forEach((cache) => {
            cache.reset();
        });
        const prompt_tokens = this.encode(prompt);
        const logits = this.forward(prompt_tokens, 0);
        let next_token = this.device.sample(this.device.softmax(logits, temperature));
        const generated_tokens = [...prompt_tokens, next_token];
        for (let step = 1; step < max_tokens; step++) {
            const logits = this.forward([next_token], generated_tokens.length - 1);
            next_token = this.device.sample(this.device.softmax(logits, temperature));
            console.log(next_token, decode([next_token]), '\n');
            generated_tokens.push(next_token);
        }
        return this.decode(generated_tokens);
    }
}
