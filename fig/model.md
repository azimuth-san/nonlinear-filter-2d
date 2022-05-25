$$
\boldsymbol{x}_{t+1} =
\begin{bmatrix}
1 & T & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & T \\
0 & 0 & 0 & 1 \\
\end{bmatrix} \boldsymbol{x}_{t}
+\begin{bmatrix}
\frac{T^2}{2} & 0 \\
T & 0 \\
0 & \frac{T^2}{2} \\
0 & T \\
\end{bmatrix}
\boldsymbol{w}_t
$$

$$
\boldsymbol{z}_{t} =
\begin{bmatrix}
\sqrt{x_t ^2 + y_t ^2}  \\
{\rm{tan}}^{^-1}(\frac{y_t}{x_t})  \\
\end{bmatrix} + \boldsymbol{v}_t
$$

$$
\boldsymbol{x}_{t} = 
\begin{bmatrix}
x_t & \dot{x_{t}} &y_t &\dot{y_{t}}
\end{bmatrix},~~
\boldsymbol{w}_{t} \in \mathbb{R}^2, ~~
\boldsymbol{v}_{t} \in \mathbb{R}^2.
$$

$$
\boldsymbol{w}_{t} \sim N(\boldsymbol{0}, Q),~~
\boldsymbol{v}_{t} \sim N(\boldsymbol{0}, R),~~
$$