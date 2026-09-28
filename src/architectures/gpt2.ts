import type { tensor } from '@/lib/tensor';
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

    constructor(metadata: _gpt2, tensors: tensor[]) {
        this.context_length = metadata['gpt2.context_length'];
        this.embedding_length = metadata['gpt2.embedding_length'];
        this.feed_forward_length = metadata['gpt2.feed_forward_length'];
        this.block_count = metadata['gpt2.block_count'];
        this.attention_head_count = metadata['gpt2.attention.head_count'];
        this.attention_layer_norm_epsilon = metadata['gpt2.attention.layer_norm_epsilon'];

        console.time('Loading tensors (sync)');
        for (const tensor of tensors) {
            console.log('loaded', tensor.name, '\t', tensor.shape);
            tensor.load();
            if (tensor.name === 'token_embd.weight') {
                this.token_embedding_tensor = tensor;
            }
            if (tensor.name === 'position_embd.weight') {
                this.position_embedding_tensor = tensor;
            }
            // console.log(tensor);
        }
        console.timeEnd('Loading tensors (sync)');
        //console.log(metadata);
        if (!this.token_embedding_tensor) {
            throw new Error('Missing token embedding tensor');
        }
        if (!this.position_embedding_tensor) {
            throw new Error('Missing position embedding tensor');
        }
    }

    private createEmbeddings(tokens: number[]): Float32Array {
        const result = new Float32Array(tokens.length * this.embedding_length);
        for (let position = 0; position < tokens.length; position++) {
            const tokenId = tokens[position];
            for (let i = 0; i < this.embedding_length; i++) {
                const tokenValue = this.token_embedding_tensor.get2D(i, tokenId);
                console.log(tokenValue);
                const positionValue = this.position_embedding_tensor.get2D(i, position);
                result[position * this.embedding_length + i] = tokenValue + positionValue;
            }
        }
        return result;
    }

    inference(tokens: number[]) {
        const embeddings = this.createEmbeddings(tokens);
        console.log(embeddings);
        return [];
    }
}
