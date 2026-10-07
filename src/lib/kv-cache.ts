export class kv_cache {
    keys: Float32Array;
    values: Float32Array;
    current_length: number = 0;
    embedding_length: number = 0;
    constructor(data: { size: number; embedding_length: number }) {
        this.keys = new Float32Array(data.size);
        this.values = new Float32Array(data.size);
        this.embedding_length = data.embedding_length;
    }

    reset() {
        this.current_length = 0;
    }

    public set(keys: Float32Array, values: Float32Array, tokenCount: number): void {
        for (let i = 0; i < tokenCount; i++) {
            const dst = (this.current_length + i) * this.embedding_length;
            const start = i * this.embedding_length;
            const end = (i + 1) * this.embedding_length;
            this.keys.set(keys.subarray(start, end), dst);
            this.values.set(values.subarray(start, end), dst);
        }
        this.current_length += tokenCount;
    }
}
