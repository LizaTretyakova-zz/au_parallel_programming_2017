#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(__global float * input, __local float * a, __local float * b, uint shift, uint n)
{
    uint gid = get_global_id(0);
    uint idx = gid * shift + shift - 1;
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
 
    if(idx < n) {
        a[lid] = b[lid] = input[gid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(idx < n) {
            if(lid > (s-1))
            {
                b[lid] = a[lid] + a[lid-s];
            }
            else
            {
                b[lid] = a[lid];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(idx < n) {
            SWAP(a,b);
        }
    }

    if(idx < n) {
        input[gid] = a[lid];
    }
}
// TODO: int --> float !!!
__kernel void propagate(__global float * input, uint shift, uint n) {
    uint gid = get_global_id(0);
    uint idx = gid * shift + shift - 1;
    uint next_shift = shift * shift;
    uint lid = get_local_id(0);

    if(idx < n && idx >= next_shift && (idx + 1) % next_shift != 0) {
        input[idx] += input[idx - idx % next_shift];
    }
}
