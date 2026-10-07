import { cpu } from '@/utils/cpu';
import type { device } from './device';

export abstract class model {
    device: device;
    constructor() {
        this.device = new cpu();
	}
    
}
