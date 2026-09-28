import type { tensor } from './tensor';

export class transformer_block {
    private tensors: Record<string, tensor> = {};
    constructor(private index: number) {}
    get(name: string) {
        if (name in this.tensors) {
            return this.tensors[name];
        }
    }
    add(tensor: tensor) {
        const parts = tensor.name.split('.');
        const tensor_name = parts.slice(2).join('.');
        this.tensors[tensor_name] = tensor;
    }
    load() {
        console.time(`Load transformer block ${this.index}`);
        for (const tensor of Object.values(this.tensors)) {
            if (tensor.status === 'cold') tensor.load();
        }
        console.timeEnd(`Load transformer block ${this.index}`);
    }
}
