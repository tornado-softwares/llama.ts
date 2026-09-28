export class Matrix {
    public data;

    public width: number = -1;
    public height: number = -1;

    constructor(data: number[][], check = false) {
        this.data = data;
        if (check) {
            if (data.length === 0) {
                throw new Error('A matrix height cannot be 0.');
            }
            this.height = data.length;
            for (const line of data) {
                if (line.length === 0) {
                    throw new Error('A matrix width cannot be 0.');
                }
                if (this.width === -1) {
                    this.width = line.length;
                } else if (line.length !== this.width) {
                    throw new Error('All matrix lines should have the same size.');
                }
            }
        } else {
            this.height = data.length;
            this.width = data[0].length;
        }
    }

    public multiply(matrix: Matrix) {
        if (this.width !== matrix.height) {
            throw new Error('Matrix dimensions are incompatible for multiplication.');
        }
        var result: number[][] = [];
        for (let i = 0; i < this.height; i++) {
            result[i] = [];
            for (let j = 0; j < matrix.width; j++) {
                let sum = 0;
                for (let k = 0; k < this.width; k++) {
                    sum += this.data[i][k] * matrix.data[k][j];
                }
                result[i][j] = sum;
            }
        }
        return new Matrix(result);
    }
}
