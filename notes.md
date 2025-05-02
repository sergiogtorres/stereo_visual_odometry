The baseline is extracted from the projection matrices. Assuming no rotation, only relative translation between cameras:

```math
  P = 
  \begin{bmatrix} 
    f_x & 0  & c_x & T_x  \\ 
    0  & f_y & c_y & T_y \\ 
    0  & 0  & 1  & T_z
  \end{bmatrix}
```

If we only have horizontal translation, then T_y, T_z are zero.

When we apply [...]