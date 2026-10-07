import util from 'node:util';

export abstract class tensor {
    public status: 'hot' | 'cold' = 'cold';
    public data: Float32Array = new Float32Array(0);
    public readonly name: string;
    public readonly dimensions: bigint[];

    constructor(name: string, dimensions: bigint[]) {
        this.name = name;
        this.dimensions = dimensions;
    }

    [util.inspect.custom]() {
        return {
            status: this.status,
            data: [...Array.from(this.data.slice(0, 10)), `...`],
            name: this.name,
            dimensions: this.dimensions,
        };
    }

    get shape(): number[] {
        return this.dimensions.map(Number);
    }

    duplicate() {
        return new external_tensor(this.dimensions, Float32Array.from(this.data));
    }

    abstract load(): void;
}

export class external_tensor extends tensor {
    constructor(dimensions: bigint[], data: Float32Array) {
        super(crypto.randomUUID(), dimensions);
        this.status = 'hot';
        this.data = data;
    }

    load() {}
}
