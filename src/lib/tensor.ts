import util from 'node:util';
import { GGML_type } from '@/utils/constants';
import { type mmap, offset } from './mmap';

export class tensor {
    public status: 'hot' | 'cold' = 'cold';
    public data: Float32Array = new Float32Array(0);
    public readonly name: string;
    public readonly dimensions: bigint[];
    public readonly type: number;
    public readonly offset: bigint;
    #file: mmap;
    #tensor_data_offset: offset;

    [util.inspect.custom]() {
        return {
            status: this.status,
            data: [...Array.from(this.data.slice(0, 10)), `...`],
            name: this.name,
            dimensions: this.dimensions,
            type: this.type,
            offset: this.offset,
        };
    }

    constructor(name: string, dimensions: bigint[], type: number, _offset: bigint, file: mmap, tensor_data_offset: offset) {
        this.name = name;
        this.dimensions = dimensions;
        this.type = type;
        this.offset = _offset;
        this.#file = file;
        this.#tensor_data_offset = new offset(tensor_data_offset.value + Number(_offset));
        tensor_data_offset;
    }

    get shape(): number[] {
        return this.dimensions.map(Number);
    }

    load() {
        const count = Number(this.dimensions.reduce((a, b) => a * b, 1n));
        const data = new Float32Array(count);
        switch (this.type) {
            case GGML_type.F32: {
                for (let i = 0; i < count; i++) data[i] = this.#file.f32(this.#tensor_data_offset);
                break;
            }
            case GGML_type.F16: {
                for (let i = 0; i < count; i++) data[i] = this.#file.f16(this.#tensor_data_offset);
                break;
            }
            case GGML_type.Q8_0: {
                const QK = 32;

                if (count % QK !== 0) {
                    throw new Error(`Invalid Q8_0 tensor size: ${count}`);
                }

                for (let i = 0; i < count; i += QK) {
                    const scale = this.#file.f16(this.#tensor_data_offset);

                    for (let j = 0; j < QK; j++) {
                        const q = this.#file.i8(this.#tensor_data_offset);
                        data[i + j] = scale * q;
                    }
                }

                break;
            }
            default: {
                throw new Error(`Tensor of type ${this.type} loader is not implemented yet. `);
            }
        }
        this.data = data;
        this.status = 'hot';
        return data;
    }
}
