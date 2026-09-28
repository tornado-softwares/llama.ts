import { mmap as _mmap } from 'bun';

export class offset {
    private _value: number;

    constructor(value: number = 0) {
        this._value = value;
    }

    get value() {
        return this._value;
    }

    increment(n: number) {
        if (n < 0) {
            throw new Error('An offset incrementation should be positive.');
        }
        this._value += n;
    }
}

export class mmap {
    private offset = new offset(0);
    private view;
    private decoder = new TextDecoder();

    constructor(file_path: string) {
        const file = _mmap(file_path);
        this.view = new DataView(file.buffer, file.byteOffset, file.byteLength);
    }

    u8(offset: offset = this.offset): number {
        const value = this.view.getUint8(offset.value);
        offset.increment(1);
        return value;
    }

    i8(offset: offset = this.offset): number {
        const value = this.view.getInt8(offset.value);
        offset.increment(1);
        return value;
    }

    u16(offset: offset = this.offset): number {
        const value = this.view.getUint16(offset.value, true);
        offset.increment(2);
        return value;
    }

    i16(offset: offset = this.offset): number {
        const value = this.view.getInt16(offset.value, true);
        offset.increment(2);
        return value;
    }

    u32(offset: offset = this.offset): number {
        const value = this.view.getUint32(offset.value, true);
        offset.increment(4);
        return value;
    }

    i32(offset: offset = this.offset): number {
        const value = this.view.getInt32(offset.value, true);
        offset.increment(4);
        return value;
    }

    u64(offset: offset = this.offset): bigint {
        const value = this.view.getBigUint64(offset.value, true);
        offset.increment(8);
        return value;
    }

    i64(offset: offset = this.offset): bigint {
        const value = this.view.getBigInt64(offset.value, true);
        offset.increment(8);
        return value;
    }

    f16(offset: offset = this.offset): number {
        const value = this.view.getFloat16(offset.value, true);
        offset.increment(2);
        return value;
    }

    f32(offset: offset = this.offset): number {
        const value = this.view.getFloat32(offset.value, true);
        offset.increment(4);
        return value;
    }

    f64(offset: offset = this.offset): number {
        const value = this.view.getFloat64(offset.value, true);
        offset.increment(8);
        return value;
    }

    bool(offset: offset = this.offset): boolean {
        return this.u8(offset) !== 0;
    }

    string(offset: offset = this.offset): string {
        const length = this.u64(offset);
        if (length > BigInt(this.view.byteLength - offset.value)) {
            throw new Error(`Invalid GGUF string length: ${length}`);
        }
        const bytes = new Uint8Array(this.view.buffer, this.view.byteOffset + offset.value, Number(length));
        offset.increment(Number(length));
        return this.decoder.decode(bytes);
    }
    array(offset: offset = this.offset): unknown[] {
        const elementType = this.u32(offset);
        const count = this.u64(offset);

        if (count > BigInt(this.view.byteLength - offset.value)) {
            throw new Error(`Invalid GGUF array count: ${count}`);
        }

        const result: unknown[] = [];

        for (let i = 0n; i < count; i++) {
            result.push(this.value(elementType, offset));
        }

        return result;
    }

    value(type: number, offset: offset = this.offset): unknown {
        switch (type) {
            case 0:
                return this.u8(offset);
            case 1:
                return this.i8(offset);
            case 2:
                return this.u16(offset);
            case 3:
                return this.i16(offset);
            case 4:
                return this.u32(offset);
            case 5:
                return this.i32(offset);
            case 6:
                return this.f32(offset);
            case 7:
                return this.bool(offset);
            case 8:
                return this.string(offset);
            case 9:
                return this.array(offset);
            case 10:
                return this.u64(offset);
            case 11:
                return this.i64(offset);
            case 12:
                return this.f64(offset);
            default:
                throw new Error(`Unsupported GGUF value type: ${type}`);
        }
    }

    get position() {
        return this.offset.value;
    }
}
