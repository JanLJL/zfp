#ifndef CUZFP_ENCODE4_CUH
#define CUZFP_ENCODE4_CUH

#include "cuZFP.h"
#include "shared.h"
#include "encode.cuh"
#include "type_info.cuh"

#define ZFP_4D_BLOCK_SIZE 256
namespace cuZFP{

template<typename Scalar> 
__device__ __host__ inline 
void gather_partial4(Scalar* q, const Scalar* p, int nx, int ny, int nz, int nw, int sx, int sy, int sz, int sw)
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
                  q[64 * w + 16 * z + 4 * y + x] = *p;
                  p += sx;
              }
              p += sy - nx * sx;
              pad_block(q + 64 * w + 16 * z + 4 * y, nx, 1);
            }
          for (x = 0; x < 4; x++)
            pad_block(q + 64 * w + 16 * z + x, ny, 4);
          p += sz - ny * sy;
        }
      for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++)
          pad_block(q + 64 * w + 4 * y + x, nz, 16);
      p += sw - nz * sz;
    }
  for (z = 0; z < 4; z++)
    for (y = 0; y < 4; y++)
      for (x = 0; x < 4; x++)
        pad_block(q + 16 * z + 4 * y + w, nw, 64);
}

template<typename Scalar> 
__device__ __host__ inline 
void gather4(Scalar* q, const Scalar* p, int sx, int sy, int sz, int sw)
{
  uint x, y, z, w;
  for (w = 0; w < 4; w++, p += sw - 4 * sz)
    for (z = 0; z < 4; z++, p += sz - 4 * sy)
      for (y = 0; y < 4; y++, p += sy - 4 * sx)
        for (x = 0; x < 4; x++, p += sx)
          *q++ = *p;
}

template<class Scalar>
__global__
void 
cudaEncode4(const uint maxbits,
           const Scalar* scalars,
           Word *stream,
           const uint4 dims,
           const int4 stride,
           const uint4 padded_dims,
           const uint tot_blocks)
{

  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is 
  // the global thread index
  const uint block_idx = blockId * blockDim.x + threadIdx.x;

  if(block_idx >= tot_blocks)
  {
    // we can't launch the exact number of blocks
    // so just exit if this isn't real
    return;
  }

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
  block.w = (block_idx / (block_dims.x * block_dims.y * block_dims.z)) * 4;

  // default strides
  ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y + (ll)block.z * stride.z + (ll)block.w * stride.w; 
  Scalar fblock[ZFP_4D_BLOCK_SIZE]; 

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
  if(block.z + 4 > dims.z) partial = true;
  if(block.w + 4 > dims.w) partial = true;
 
  if(partial) 
  {
    const uint nx = block.x + 4u > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4u > dims.y ? dims.y - block.y : 4;
    const uint nz = block.z + 4u > dims.z ? dims.z - block.z : 4;
    const uint nw = block.w + 4u > dims.w ? dims.w - block.w : 4;

    gather_partial4(fblock, scalars + offset, nx, ny, nz, nw, stride.x, stride.y, stride.z, stride.w);

  }
  else
  {
    gather4(fblock, scalars + offset, stride.x, stride.y, stride.z, stride.w);
  }
  zfp_encode_block<Scalar, ZFP_4D_BLOCK_SIZE>(fblock, maxbits, block_idx, stream);  

}

//
// Launch the encode kernel
//
template<class Scalar>
size_t encode4launch(uint4 dims, 
                     int4 stride,
                     const Scalar *d_data,
                     Word *stream,
                     const int maxbits)
{

  const int cuda_block_size = 512;
  dim3 block_size = dim3(cuda_block_size, 1, 1);

  uint4 zfp_pad(dims); 
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;
  if(zfp_pad.z % 4 != 0) zfp_pad.z += 4 - dims.z % 4;
  if(zfp_pad.w % 4 != 0) zfp_pad.w += 4 - dims.w % 4;

  const uint zfp_blocks = (zfp_pad.x * zfp_pad.y * zfp_pad.z * zfp_pad.w) / 256; 

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

  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

  size_t stream_bytes = calc_device_mem4d(zfp_pad, maxbits);
  //ensure we start with 0s
  cudaMemset(stream, 0, stream_bytes);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  cudaEncode4<Scalar> <<<grid_size, block_size>>>
    (maxbits,
     d_data,
     stream,
     dims,
     stride,
     zfp_pad,
     zfp_blocks);

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
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("# encode4 rate: %.2f (GB / sec) \n", rate);
#endif
  return stream_bytes;
}

//
// Just pass the raw pointer to the "real" encode
//
template<class Scalar>
size_t encode4(uint4 dims, 
              int4 stride,
              Scalar *d_data,
              Word *stream,
              const int bits_per_block)
{
  return encode4launch<Scalar>(dims, stride, d_data, stream, bits_per_block);
}

}
#endif
