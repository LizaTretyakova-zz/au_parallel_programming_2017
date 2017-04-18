__kernel void matrix_conv(__global float * a, __global float * b, __global float * c, int n, int m)
{
   int row = get_global_id(0);
   int col = get_global_id(1);

   if (row >= n || col >= n)
      return;

   int hm = (m - 1) / 2;
   float sum = 0;

    for(int k = -hm; k <= hm; ++k) {
        for(int l = -hm; l <= hm; ++l) {
            if(row + k >= 0 && row + k < n
                && col + l >= 0 && col + l < n) {
                sum += a[(row + k) * n + (col + l)]
                        * b[(k + hm) * m + (l + hm)];
            }
        }
    }

   c[row * n + col] = sum;
}
