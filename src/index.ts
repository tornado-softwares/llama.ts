import { Matrix } from '@/lib/matrix';
import { gpt2 } from './architectures/gpt2';
import { GGUF } from './lib/gguf';

const a = new Matrix([
    [1, 2],
    [3, 4],
]);

const b = new Matrix([
    [5, 6],
    [7, 8],
]);

const c = new Matrix([[1, 5]], true);

console.log(a.multiply(b));
console.log(b.multiply(b));

console.log(a.multiply(a));
console.log(b.multiply(b));

const gguf = new GGUF('/home/tornado-softwares/Bureau/llama.ts/models/gpt2.Q8_0.gguf');
const tokens: number[] = [40313, 318, 257, 1295, 810];

if (gguf.metadata['general.architecture'] === 'gpt2') {
    const model = new gpt2(gguf.metadata, gguf.tensors);
    console.log(model.inference(tokens));
}
