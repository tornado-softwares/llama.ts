import type { metadata } from '@/types/gguf';
import { mmap, offset } from './mmap';
import { tensor } from './tensor';
import { transformer_block } from './transformer-block';

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
            const new_tensor = new tensor(info.name, info.dimensions, info.type, info.offset, file, tensor_data_offset);
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
