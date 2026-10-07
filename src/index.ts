import { gpt2 } from './architectures/gpt2';
import { GGUF } from './lib/gguf';

const gguf = new GGUF('/home/tornado-softwares/Bureau/llama.ts/models/gpt2.Q8_0.gguf');

if (gguf.metadata['general.architecture'] === 'gpt2') {
    const model = new gpt2(gguf.metadata, gguf.tensors, gguf.transformer_blocks);
    const generated = model.inference({
        prompt: 'Yesterday, a tornado',
        max_tokens: 100,
        temperature: 0.7,
    });
    console.log(generated);
}
