export interface gpuU32Inputs {
  u32Inputs: Uint32Array;
  individualInputSize: number;
}

export const bigIntsToU16Array = (beBigInts: bigint[]): Uint16Array => {
  const intsAs16s = beBigInts.map((bigInt) => bigIntToU16Array(bigInt));
  const u16Array = new Uint16Array(beBigInts.length * 16);
  intsAs16s.forEach((intAs16, index) => {
    u16Array.set(intAs16, index * 16);
  });
  return u16Array;
};

export const bigIntToU16Array = (beBigInt: bigint): Uint16Array => {
  const numBits = 256;
  const bitsPerElement = 16;
  const numElements = numBits / bitsPerElement;
  const u16Array = new Uint16Array(numElements);
  const mask = (BigInt(1) << BigInt(bitsPerElement)) - BigInt(1); // Create a mask for the lower 32 bits

  let tempBigInt = beBigInt;
  for (let i = numElements - 1; i >= 0; i--) {
    u16Array[i] = Number(tempBigInt & mask); // Extract the lower 32 bits
    tempBigInt >>= BigInt(bitsPerElement); // Right-shift the remaining bits
  }

  return u16Array;
};

export const flattenU32s = (u32Arrays: Uint32Array[]): Uint32Array => {
  const flattenedU32s = new Uint32Array(u32Arrays.length * u32Arrays[0].length);
  u32Arrays.forEach((u32Array, index) => {
    flattenedU32s.set(u32Array, index * u32Array.length);
  });
  return flattenedU32s;
};

// assume bigints are big endian 256-bit integers
export const bigIntsToU32Array = (beBigInts: bigint[]): Uint32Array => {
  const intsAs32s = beBigInts.map((bigInt) => bigIntToU32Array(bigInt));
  const u32Array = new Uint32Array(beBigInts.length * 8);
  intsAs32s.forEach((intAs32, index) => {
    u32Array.set(intAs32, index * 8);
  });
  return u32Array;
};

export const bigIntToU32Array = (beBigInt: bigint): Uint32Array => {
  const numBits = 256;
  const bitsPerElement = 32;
  const numElements = numBits / bitsPerElement;
  const u32Array = new Uint32Array(numElements);
  const mask = (BigInt(1) << BigInt(bitsPerElement)) - BigInt(1); // Create a mask for the lower 32 bits

  let tempBigInt = beBigInt;
  for (let i = numElements - 1; i >= 0; i--) {
    u32Array[i] = Number(tempBigInt & mask); // Extract the lower 32 bits
    tempBigInt >>= BigInt(bitsPerElement); // Right-shift the remaining bits
  }

  return u32Array;
};

export const u32ArrayToBigInts = (u32Array: Uint32Array): bigint[] => {
  const bigInts = [];
  const chunkSize = 8;
  const bitsPerElement = 32;

  for (let i = 0; i < u32Array.length; i += chunkSize) {
    let bigInt = BigInt(0);
    for (let j = 0; j < chunkSize; j++) {
      if (i + j >= u32Array.length) break; // Avoid out-of-bounds access
      const u32 = BigInt(u32Array[i + j]);
      bigInt |= u32 << (BigInt(chunkSize - 1 - j) * BigInt(bitsPerElement));
    }
    bigInts.push(bigInt);
  }

  return bigInts;
};
