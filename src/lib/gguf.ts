import type { metadata } from '@/types/gguf';
import { GGML_type } from '@/utils/constants';
import { mmap, offset } from './mmap';
import { tensor } from './tensor';
import { transformer_block } from './transformer-block';

export class gguf_tensor extends tensor {
    public readonly offset: bigint;
    public readonly type: number;
    #file: mmap;
    #tensor_data_offset: offset;

    constructor(name: string, dimensions: bigint[], type: number, _offset: bigint, file: mmap, tensor_data_offset: offset) {
        super(name, dimensions);
        this.type = type;
        this.offset = _offset;
        this.#file = file;
        this.#tensor_data_offset = new offset(tensor_data_offset.value + Number(_offset));
        tensor_data_offset;
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
        console.log('Loaded', this.name, '\t', this.shape);
        return data;
    }
}

export class GGUF {
    public version;
    public tensor_count;
    public metadata_kv_count;

    public metadata: metadata = {} as any;
    public tensors: tensor[] = [];

    public transformer_blocks: transformer_block[] = [];

    constructor(file_path: string) {
        const file = new mmap(file_path);

        const magic = file.u32();
        if (magic !== 0x46554747) {
            throw new Error(`${file_path} is not a GGUF.`);
        }

        this.version = file.u32();
        this.tensor_count = file.u64();
        this.metadata_kv_count = file.u64();

        for (let i = 0; i < this.metadata_kv_count; i++) {
            const key = file.string();
            const value_type = file.u32();
            const value = file.value(value_type);
            this.metadata[key] = value;
        }

        const tensor_infos: {
            name: string;
            dimensions: bigint[];
            type: number;
            offset: bigint;
        }[] = [];

        for (let i = 0; i < this.tensor_count; i++) {
            const name = file.string();
            const n_dimensions = file.u32();
            const dimensions: bigint[] = [];
            for (let j = 0; j < n_dimensions; j++) dimensions.push(file.u64());
            const type = file.u32();
            const offset = file.u64();
            tensor_infos.push({ name, dimensions, type, offset });
        }

        const alignment = this.metadata['general.alignment'] ? BigInt(this.metadata['general.alignment']) : 32n;
        const tensor_data_offset = new offset(Number(align_offset(BigInt(file.position), alignment)));
        for (const info of tensor_infos) {
            const new_tensor = new gguf_tensor(info.name, info.dimensions, info.type, info.offset, file, tensor_data_offset);
            this.tensors.push(new_tensor);
            if (new_tensor.name.startsWith('blk.')) {
                const parts = new_tensor.name.split('.');
                const block_index = parseInt(parts[1], 10);
                if (!this.transformer_blocks[block_index]) {
                    this.transformer_blocks[block_index] = new transformer_block(block_index);
                }
                this.transformer_blocks[block_index].add(new_tensor);
            }
        }
    }
}

/* https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 * uint64_t align_offset(uint64_t offset) {
 *     return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
 * }
 */
function align_offset(offset: bigint, alignment: bigint): bigint {
    return offset + ((alignment - (offset % alignment)) % alignment);
}
