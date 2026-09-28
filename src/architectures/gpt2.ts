import type { tensor } from '@/lib/tensor';
import type { transformer_block } from '@/lib/transformer-block';
import type { gpt2 as _gpt2 } from '@/types/gguf';

export class gpt2 {
    private context_length;
    private embedding_length;
    private feed_forward_length;
    private block_count;
    private attention_head_count;
    private attention_layer_norm_epsilon;

    private token_embedding_tensor!: tensor;
    private position_embedding_tensor!: tensor;
    private transformer_blocks: transformer_block[] = [];

    constructor(metadata: _gpt2, tensors: tensor[], transformer_blocks: transformer_block[]) {
        this.context_length = metadata['gpt2.context_length'];
        this.embedding_length = metadata['gpt2.embedding_length'];
        this.feed_forward_length = metadata['gpt2.feed_forward_length'];
        this.block_count = metadata['gpt2.block_count'];
        this.attention_head_count = metadata['gpt2.attention.head_count'];
        this.attention_layer_norm_epsilon = metadata['gpt2.attention.layer_norm_epsilon'];

        this.transformer_blocks = transformer_blocks;

        for (const tensor of tensors) {
            // console.log('loaded', tensor.name, '\t', tensor.shape);
            if (tensor.name === 'token_embd.weight') {
                this.token_embedding_tensor = tensor;
            }
            if (tensor.name === 'position_embd.weight') {
                this.position_embedding_tensor = tensor;
            }
        }

        if (!this.token_embedding_tensor) {
            throw new Error('Missing token embedding tensor');
        }
        if (!this.position_embedding_tensor) {
            throw new Error('Missing position embedding tensor');
        }
    }

    private get2D(tensor: tensor, i: number, j: number): number {
        const [dim0, dim1] = tensor.shape;
        if (i < 0 || i >= dim0) {
            throw new RangeError(`i=${i}, dim0=${dim0}`);
        }
        if (j < 0 || j >= dim1) {
            throw new RangeError(`j=${j}, dim1=${dim1}`);
        }
        return tensor.data[i + dim0 * j];
    }

    mean(vector: Float32Array) {
        let sum = 0;
        for (const value of vector) {
            sum += value;
        }
        return sum / vector.length;
    }

    inference(tokens: number[]) {
        if (this.token_embedding_tensor.status === 'cold') this.token_embedding_tensor.load();
        if (this.position_embedding_tensor.status === 'cold') this.position_embedding_tensor.load();

        const token_embeddings = new Float32Array(tokens.length * this.embedding_length);
        for (let position = 0; position < tokens.length; position++) {
            const token_id = tokens[position];
            for (let i = 0; i < this.embedding_length; i++) {
                token_embeddings[position * this.embedding_length + i] =
                    this.get2D(this.token_embedding_tensor, i, token_id) + this.get2D(this.position_embedding_tensor, i, position);
            }
        }

        // console.log(this.transformer_blocks);

        this.transformer_blocks[0].load();

        /*for (let position = 0; position < tokens.length; position++) {
            const token_embedding = token_embeddings.subarray(position * this.embedding_length, (position + 1) * this.embedding_length);
            const token_mean = this.mean(token_embedding);
        }
        */

        // console.log(token_embeddings);
        return [];
    }
}
