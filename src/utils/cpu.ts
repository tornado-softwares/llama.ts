import { device } from '@/lib/device';
import { external_tensor, type tensor } from '@/lib/tensor';

export class cpu extends device {
    mean(data: Float32Array, unit_size: number, y_offset: number = 0): number {
        let sum = 0;
        for (let x_offset = 0; x_offset < unit_size; x_offset++) {
            sum += data[y_offset + x_offset];
        }
        return sum / unit_size;
    }

    variance(data: Float32Array, unit_size: number, mean: number, y_offset: number = 0): number {
        let sum = 0;
        for (let x_offset = 0; x_offset < unit_size; x_offset++) {
            const diff = data[y_offset + x_offset] - mean;
            sum += diff * diff;
        }
        return sum / unit_size;
    }

    standard_deviation(variance: number, epsilon: number = 0): number {
        return Math.sqrt(variance + epsilon);
    }

    linear({ input, weight, bias }: { input: tensor; weight: tensor; bias?: tensor }): external_tensor {
        const [in_dim, n_tokens] = input.shape;
        const [w_in, out_dim] = weight.shape;

        if (in_dim !== w_in) {
            throw new Error(`linear: input dim0=${in_dim} != weight dim0=${w_in}`);
        }
        if (bias && bias.shape[0] !== out_dim) {
            throw new Error(`linear: bias size=${bias.shape[0]} != out_dim=${out_dim}`);
        }

        const out = new Float32Array(out_dim * n_tokens);

        for (let t = 0; t < n_tokens; t++) {
            const x_off = t * in_dim;
            const o_off = t * out_dim;
            for (let col = 0; col < out_dim; col++) {
                const w_off = col * in_dim;
                let acc = 0;
                for (let row = 0; row < in_dim; row++) {
                    acc += input.data[x_off + row] * weight.data[w_off + row];
                }
                out[o_off + col] = acc + (bias ? bias.data[col] : 0);
            }
        }

        return new external_tensor([BigInt(out_dim), BigInt(n_tokens)], out);
    }

    split_columns(x: tensor, parts: number): external_tensor[] {
        const [dim0, n_tokens] = x.shape;
        if (dim0 % parts !== 0) {
            throw new Error(`split_columns: dim0=${dim0} not divisible by ${parts}`);
        }
        const part_dim = dim0 / parts;
        return Array.from({ length: parts }, (_, p) => {
            const out = new Float32Array(part_dim * n_tokens);
            for (let t = 0; t < n_tokens; t++) {
                const start = t * dim0 + p * part_dim;
                out.set(x.data.subarray(start, start + part_dim), t * part_dim);
            }
            return new external_tensor([BigInt(part_dim), BigInt(n_tokens)], out);
        });
    }

    layer_norm({ input, weight, bias, epsilon }: { input: tensor; weight: tensor; bias: tensor; epsilon: number }): external_tensor {
        const [hidden_size, n_tokens] = input.shape;
        if (weight.shape[0] !== hidden_size || bias.shape[0] !== hidden_size) {
            throw new Error(`layer_norm: dim=${hidden_size} != weight=${weight.shape[0]} / bias=${bias.shape[0]}`);
        }
        const out = new Float32Array(hidden_size * n_tokens);
        for (let t = 0; t < n_tokens; t++) {
            const y_offset = t * hidden_size;

            const mean = this.mean(input.data, hidden_size, y_offset); //                           |
            const variance = this.variance(input.data, hidden_size, mean, y_offset); //             |-->  mean(μ) -> variance(σ²) -> standard_deviation(σ)
            const standard_deviation = this.standard_deviation(variance, epsilon); //                |
            //                                                              ↑
            // 	  μ : 'mu'   ϵ : 'epsilon'	σ : 'sigma'	      The standard_deviation calculation is implemented as √(σ² + ϵ) Why ϵ ?
            // 												  Because running only √(σ²) on a GPU can produce NaN when variance is close to 0
            //
            for (let x_offset = 0; x_offset < hidden_size; x_offset++) {
                const normalized = (input.data[y_offset + x_offset] - mean) / standard_deviation;
                //                                     ↑                                 ↑
                // 								    (x - μ)			   /	         √(σ² + ϵ)
                //
                out[y_offset + x_offset] = normalized * weight.data[x_offset] + bias.data[x_offset];
                //                             ↓                  ↓                       ↓
                //                          z-score     *    learned weight     +    learned shift
                //
                // bonus:    You maybe saw the form of affine function, in layer norm we can speak about affine parameters..
            }
        }

        return new external_tensor([BigInt(hidden_size), BigInt(n_tokens)], out);
    }

    add(a: tensor, b: tensor): external_tensor {
        if (a.data.length !== b.data.length) {
            throw new Error(`add: size mismatch ${a.data.length} != ${b.data.length}`);
        }

        const out = new Float32Array(a.data.length);
        for (let i = 0; i < out.length; i++) {
            out[i] = a.data[i] + b.data[i];
        }

        return new external_tensor(a.dimensions, out);
    }

    gelu(x: tensor): external_tensor {
        const out = new Float32Array(x.data.length);
        const c = Math.sqrt(2 / Math.PI);

        for (let i = 0; i < out.length; i++) {
            const v = x.data[i];
            out[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v ** 3)));
        }

        return new external_tensor(x.dimensions, out);
    }

    embed({ token_embedding, position_embedding, tokens, start_position }: { token_embedding: tensor; position_embedding: tensor; tokens: number[]; start_position: number }): external_tensor {
        const [hidden_size, vocab_size] = token_embedding.shape;
        const [pos_hidden, max_positions] = position_embedding.shape;

        if (hidden_size !== pos_hidden) {
            throw new Error(`embed: token hidden=${hidden_size} != position hidden=${pos_hidden}`);
        }

        const out = new Float32Array(tokens.length * hidden_size);

        for (let token_index = 0; token_index < tokens.length; token_index++) {
            const token_id = tokens[token_index];
            const position = start_position + token_index;

            if (token_id < 0 || token_id >= vocab_size) throw new RangeError(`embed: token_id=${token_id}, vocab=${vocab_size}`);
            if (position >= max_positions) throw new RangeError(`embed: position=${position}, max=${max_positions}`);

            const tok_y_offset = token_id * hidden_size;
            const pos_y_offset = position * hidden_size;
            const out_y_offset = token_index * hidden_size;

            for (let x_offset = 0; x_offset < hidden_size; x_offset++) {
                const embedding = token_embedding.data[tok_y_offset + x_offset]; // -> Token information
                const position = position_embedding.data[pos_y_offset + x_offset]; // -> Token position
                out[out_y_offset + x_offset] = embedding + position;
                // 	             |                       ↓
                //               |                    In the hidden state, we build a vector containing each token information & position (combined)
                //			     ↓
                // 		The 2 dimensional tensors are flat, so we need to access a value data[y][x] with offset sums.
                //      Ex: gpt-2
                // 		The first token data is in the range [ 0; 767 ]
                // 		The second token data is in the range [ 768; 1535 ]
                // 		The n token data is in the range [ n * 768; n * 768 + 767 ]
            }
        }

        return new external_tensor([BigInt(hidden_size), BigInt(tokens.length)], out);
    }

    take_token(x: tensor, token_index: number): external_tensor {
        const [dim, n_tokens] = x.shape;
        if (token_index < 0 || token_index >= n_tokens) {
            throw new RangeError(`take_token: index=${token_index}, n_tokens=${n_tokens}`);
        }

        const out = x.data.slice(token_index * dim, (token_index + 1) * dim);
        return new external_tensor([BigInt(dim), 1n], out);
    }

    causal_attention({
        query,
        keys,
        values,
        n_tokens,
        total_len,
        start_position,
        num_heads,
        hidden_size,
    }: {
        query: Float32Array;
        keys: Float32Array;
        values: Float32Array;
        n_tokens: number;
        total_len: number;
        start_position: number;
        num_heads: number;
        hidden_size: number;
    }): external_tensor {
        const head_dim = hidden_size / num_heads;
        const scale = 1 / Math.sqrt(head_dim);
        const out = new Float32Array(n_tokens * hidden_size);
        const scores = new Float32Array(total_len);
        for (let h = 0; h < num_heads; h++) {
            const head_off = h * head_dim;

            for (let i = 0; i < n_tokens; i++) {
                const q_off = i * hidden_size + head_off;
                const last = Math.min(start_position + i, total_len - 1);

                let max = -Infinity;
                for (let j = 0; j <= last; j++) {
                    const k_off = j * hidden_size + head_off;
                    let dot = 0;
                    for (let d = 0; d < head_dim; d++) {
                        dot += query[q_off + d] * keys[k_off + d];
                    }
                    scores[j] = dot * scale;
                    if (scores[j] > max) max = scores[j];
                }

                let sum = 0;
                for (let j = 0; j <= last; j++) {
                    scores[j] = Math.exp(scores[j] - max);
                    sum += scores[j];
                }
                for (let j = 0; j <= last; j++) {
                    scores[j] /= sum;
                }

                for (let d = 0; d < head_dim; d++) {
                    let acc = 0;
                    for (let j = 0; j <= last; j++) {
                        acc += scores[j] * values[j * hidden_size + head_off + d];
                    }
                    out[q_off + d] = acc;
                }
            }
        }

        return new external_tensor([BigInt(hidden_size), BigInt(n_tokens)], out);
    }

    sample(probs: tensor): number {
        const r = Math.random();
        let cumulative = 0;

        for (let i = 0; i < probs.data.length; i++) {
            cumulative += probs.data[i];

            if (r < cumulative) {
                return i;
            }
        }

        return probs.data.length - 1;
    }

    argmax(tensor: tensor): number {
        let best = 0;
        for (let i = 1; i < tensor.data.length; i++) {
            if (tensor.data[i] > tensor.data[best]) best = i;
        }
        return best;
    }
    softmax(_tensor: tensor, temperature: number = 1): tensor {
        const tensor = _tensor.duplicate();

        let max = -Infinity;

        for (let i = 0; i < tensor.data.length; i++) {
            max = Math.max(max, tensor.data[i]);
        }

        let sum = 0;

        for (let i = 0; i < tensor.data.length; i++) {
            tensor.data[i] = Math.exp((tensor.data[i] - max) / temperature);
            sum += tensor.data[i];
        }

        for (let i = 0; i < tensor.data.length; i++) {
            tensor.data[i] /= sum;
        }

        return tensor;
    }
}
