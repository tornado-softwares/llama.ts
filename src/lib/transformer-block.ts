import type { tensor } from './tensor';

export class transformer_block {
    public status: 'hot' | 'cold' = 'cold';
    private tensors: Record<string, tensor> = {};

    constructor(private index: number) {}

    has(name: string) {
        if (name in this.tensors) {
            return true;
        }
        return false;
    }
    get(name: string) {
        if (name in this.tensors) {
            return this.tensors[name];
        }
        throw new Error(`missing ${name} tensor in block ${this.index}`);
    }
    add(tensor: tensor) {
        const parts = tensor.name.split('.');
        const tensor_name = parts.slice(2).join('.');
        this.tensors[tensor_name] = tensor;
    }
    load() {
        if (this.status === 'hot') return;
        console.time(`Loaded transformer block ${this.index}`);
        for (const tensor of Object.values(this.tensors)) {
            if (tensor.status === 'cold') tensor.load();
        }
        console.timeEnd(`Loaded transformer block ${this.index}`);
        this.status = 'hot';
    }
}
