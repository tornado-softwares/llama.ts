import type { external_tensor, tensor } from '@/lib/tensor';

export abstract class device {
    abstract mean(data: Float32Array, unit_size: number, y_offset?: number): number;
    abstract variance(data: Float32Array, unit_size: number, mean: number, y_offset?: number): number;
    abstract standard_deviation(variance: number, epsilon?: number): number;
    abstract linear(data: { input: tensor; weight: tensor; bias?: tensor }): external_tensor;
    abstract split_columns(x: tensor, parts: number): external_tensor[];
    abstract layer_norm(data: { input: tensor; weight: tensor; bias: tensor; epsilon: number }): external_tensor;
    abstract add(a: tensor, b: tensor): external_tensor;
    abstract gelu(x: tensor): external_tensor;
    abstract embed(data: { token_embedding: tensor; position_embedding: tensor; tokens: number[]; start_position: number }): external_tensor;
    abstract take_token(x: tensor, token_index: number): external_tensor;
    abstract causal_attention(data: { query: Float32Array; keys: Float32Array; values: Float32Array; n_tokens: number; total_len: number; start_position: number; num_heads: number; hidden_size: number }): external_tensor;
    abstract argmax(tensor: tensor): number;
    abstract sample(tensor: tensor): number;
    abstract softmax(tensor: tensor, temperature: number): external_tensor;
}
