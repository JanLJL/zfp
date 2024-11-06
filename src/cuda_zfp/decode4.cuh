#ifndef CUZFP_DECODE4_CUH
#define CUZFP_DECODE4_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {

template<typename Scalar> 
__device__ __host__ inline 
void scatter_partial4(const Scalar* q, Scalar* p, int nx, int ny, int nz, int nw, int sx, int sy, int sz, int sw)
{
  uint x, y, z, w;
  for (w = 0; w < 4; w++)
    if (w < nw) {
      for (z = 0; z < 4; z++)
        if (z < nz) {
          for (y = 0; y < 4; y++)
            if (y < ny) {
              for (x = 0; x < 4; x++)
                if (x < nx) {
                  *p = q[64 * w + 16 * z + 4 * y + x];
                  p += sx;
                }
              p += sy - nx * sx;
            }
          p += sz - ny * sy;
        }
      p += sw - nz * sz;
    }
}

template<typename Scalar> 
__device__ __host__ inline 
void scatter4(const Scalar* q, Scalar* p, int sx, int sy, int sz, int sw)
{
  uint x, y, z, w;
  for (w = 0; w < 4; w++, p += sw - 4 * sz)
    for (z = 0; z < 4; z++, p += sz - 4 * sy)
      for (y = 0; y < 4; y++, p += sy - 4 * sx)
        for (x = 0; x < 4; x++, p += sx)
          *p = *q++;
}


template<class Scalar, int BlockSize>
__global__
void
cudaDecode4(Word *blocks,
            Scalar *out,
            const uint4 dims,
            const int4 stride,
            const uint4 padded_dims,
            uint maxbits)
{
  
  typedef unsigned long long int ull;
  typedef long long int ll;

  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;
  // each thread gets a block so the block index is 
  // the global thread index
  const ull block_idx = blockId * blockDim.x + threadIdx.x;
  
  const int total_blocks = (padded_dims.x * padded_dims.y * padded_dims.z * padded_dims.w) / 256; 
  //std::cout << "[DECOMP] total blocks: " << total_blocks << std::endl;

  if(block_idx >= total_blocks) 
  {
    return;
  }

  BlockReader<BlockSize> reader(blocks, maxbits, block_idx, total_blocks);
 
  Scalar result[BlockSize];
  memset(result, 0, sizeof(Scalar) * BlockSize);

  zfp_decode<Scalar,BlockSize>(reader, result, maxbits);

  // logical block dims
  uint4 block_dims;
  block_dims.x = padded_dims.x >> 2; 
  block_dims.y = padded_dims.y >> 2; 
  block_dims.z = padded_dims.z >> 2; 
  block_dims.w = padded_dims.w >> 2;
  // logical pos in 4d array
  uint4 block;
  block.x = (block_idx % block_dims.x) * 4; 
  block.y = ((block_idx / block_dims.x) % block_dims.y) * 4; 
  block.z = ((block_idx / (block_dims.x * block_dims.y)) % block_dims.z) * 4; 
  block.w = ((block_idx / (block_dims.x * block_dims.y * block_dims.z) %block_dims.w)) * 4;
  
  // default strides
  const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y + (ll)block.z * stride.z + (ll)block.w * stride.w; 

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
  if(block.z + 4 > dims.z) partial = true;
  if(block.w + 4 > dims.w) partial = true;
  //std::cout << "Partial=" << partial << std::endl;
  if(partial)
  {
    const uint nx = block.x + 4u > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4u > dims.y ? dims.y - block.y : 4;
    const uint nz = block.z + 4u > dims.z ? dims.z - block.z : 4;
    const uint nw = block.w + 4u > dims.w ? dims.w - block.w : 4;

    scatter_partial4(result, out + offset, nx, ny, nz, nw, stride.x, stride.y, stride.z, stride.w);
  }
  else
  {
    scatter4(result, out + offset, stride.x, stride.y, stride.z, stride.w);
  }
}

template<class Scalar>
size_t decode4launch(uint4 dims, 
                     int4 stride,
                     Word *stream,
                     Scalar *d_data,
                     uint maxbits)
{
  const int cuda_block_size = 128;
  dim3 block_size;
  block_size = dim3(cuda_block_size, 1, 1);

  uint4 zfp_pad(dims); 
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;
  if(zfp_pad.z % 4 != 0) zfp_pad.z += 4 - dims.z % 4;
  if(zfp_pad.w % 4 != 0) zfp_pad.w += 4 - dims.w % 4;

  const int zfp_blocks = (zfp_pad.x * zfp_pad.y * zfp_pad.z * zfp_pad.w) / 256; 

  
  //
  // we need to ensure that we launch a multiple of the 
  // cuda block size
  //
  int block_pad = 0; 
  if(zfp_blocks % cuda_block_size != 0)
  {
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size; 
  }

  size_t total_blocks = block_pad + zfp_blocks;
  size_t stream_bytes = calc_device_mem4d(zfp_pad, maxbits);

  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

#ifdef CUDA_ZFP_RATE_PRINT
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
#endif

  cudaDecode4<Scalar, 256> << < grid_size, block_size >> >
    (stream,
		 d_data,
     dims,
     stride,
     zfp_pad,
     maxbits);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds / 1000.f;
  float rate = (float(dims.x * dims.y * dims.z * dims.w) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode4 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif

  return stream_bytes;
}

template<class Scalar>
size_t decode4(uint4 dims, 
               int4 stride,
               Word  *stream,
               Scalar *d_data,
               uint maxbits)
{
	return decode4launch<Scalar>(dims, stride, stream, d_data, maxbits);
}

} // namespace cuZFP

#endif
