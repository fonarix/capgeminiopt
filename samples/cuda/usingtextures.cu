
__global__ void kernel(cudaTextureObject_t tex, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float value = tex2D<float>(tex, x, y);  // Получаем значение из текстурной памяти
        output[y * width + x] = value * 2.0f;  // Пример работы с данными
    }
}



int main()
{

cudaTextureObject_t tex;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
cudaArray_t cuArray;

// Allocate texture memory
cudaMallocArray(&cuArray, &channelDesc, width, height);

// Copy data to texture
cudaMemcpyToArray(cuArray, 0, 0, h_data, width * height * sizeof(float), cudaMemcpyHostToDevice);

// Tesure memory description
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;

cudaTextureDesc texDesc = {};
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModePoint;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = false;


cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);


kernel<<<>>>

cudaDestroyTextureObject(tex);
cudaFreeArray(cuArray);



}