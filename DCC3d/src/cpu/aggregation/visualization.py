import matplotlib.pyplot as plt
import numpy as np

# 尝试导入 plotly，如果没有安装则跳过
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PointCloudVisualizer:
    def __init__(self, coords, indices):
        """
        coords: (N, 3) 原始点云坐标
        indices: (N, n) 选点算法输出的邻居索引
        """
        self.coords = coords
        self.indices = indices
        self.N, self.n_sample = indices.shape

    def _get_local_data(self, center_idx):
        """获取指定中心点的局部数据"""
        center_pos = self.coords[center_idx]
        neighbor_indices = self.indices[center_idx]
        neighbor_pos = self.coords[neighbor_indices]
        return center_pos, neighbor_pos

    def plot_static(self, center_idx=0, radius=None, title=None):
        """
        使用 Matplotlib 绘制静态图
        center_idx: 要可视化的中心点索引
        radius: 如果是 Ball Query，传入半径画圆
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        center_pos, neighbor_pos = self._get_local_data(center_idx)

        # 1. 绘制背景点云 (为了性能，只随机采样部分背景点，或者画得很淡)
        # 如果点太多，只画 1000 个
        bg_indices = np.random.choice(self.N, min(self.N, 1000), replace=False)
        ax.scatter(
            self.coords[bg_indices, 0],
            self.coords[bg_indices, 1],
            self.coords[bg_indices, 2],
            c="lightgray",
            s=1,
            alpha=0.3,
            label="Background",
        )

        # 2. 绘制邻居点 (绿色)
        ax.scatter(
            neighbor_pos[:, 0],
            neighbor_pos[:, 1],
            neighbor_pos[:, 2],
            c="green",
            s=20,
            label="Neighbors",
        )

        # 3. 绘制中心点 (红色)
        ax.scatter(
            center_pos[0],
            center_pos[1],
            center_pos[2],
            c="red",
            s=50,
            marker="*",
            label="Center",
        )

        # 4. 绘制连线
        for nb in neighbor_pos:
            ax.plot(
                [center_pos[0], nb[0]],
                [center_pos[1], nb[1]],
                [center_pos[2], nb[2]],
                "k--",
                alpha=0.5,
                linewidth=0.5,
            )

        # 5. (可选) 绘制球体线框
        if radius is not None:
            self._draw_wireframe_sphere(ax, center_pos, radius)

        ax.set_title(title if title else f"Selection Visualization (Idx: {center_idx})")
        ax.legend()
        plt.show()

    def plot_interactive(self, center_idx=0, radius=None, filename=None):
        """
        使用 Plotly 绘制交互式图形 (HTML)
        """
        if not PLOTLY_AVAILABLE:
            print(
                "Error: Plotly library not found. Please install via 'pip install plotly'"
            )
            return

        center_pos, neighbor_pos = self._get_local_data(center_idx)

        data = []

        # 1. 背景点云 (灰色小点)
        data.append(
            go.Scatter3d(
                x=self.coords[:, 0],
                y=self.coords[:, 1],
                z=self.coords[:, 2],
                mode="markers",
                marker=dict(size=1, color="lightgray", opacity=0.3),
                name="Cloud",
            )
        )

        # 2. 邻居点 (绿色)
        data.append(
            go.Scatter3d(
                x=neighbor_pos[:, 0],
                y=neighbor_pos[:, 1],
                z=neighbor_pos[:, 2],
                mode="markers",
                marker=dict(size=4, color="green"),
                name="Neighbors",
            )
        )

        # 3. 中心点 (红色大点)
        data.append(
            go.Scatter3d(
                x=[center_pos[0]],
                y=[center_pos[1]],
                z=[center_pos[2]],
                mode="markers",
                marker=dict(size=6, color="red", symbol="diamond"),
                name="Center",
            )
        )

        # 4. 连线
        # Plotly 画线需要构造 x, y, z 列表，并在每段线之间插入 None
        x_lines, y_lines, z_lines = [], [], []
        for nb in neighbor_pos:
            x_lines.extend([center_pos[0], nb[0], None])
            y_lines.extend([center_pos[1], nb[1], None])
            z_lines.extend([center_pos[2], nb[2], None])

        data.append(
            go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode="lines",
                line=dict(color="black", width=2),
                name="Connections",
            )
        )

        # 5. (可选) 半径球体
        if radius is not None:
            # 简单生成一个球的散点来表示边界，或者使用 Mesh3d
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = center_pos[0] + radius * np.cos(u) * np.sin(v)
            y = center_pos[1] + radius * np.sin(u) * np.sin(v)
            z = center_pos[2] + radius * np.cos(v)
            data.append(
                go.Mesh3d(
                    x=x.flatten(),
                    y=y.flatten(),
                    z=z.flatten(),
                    color="blue",
                    opacity=0.1,
                    alphahull=0,
                    name="Query Radius",
                )
            )

        layout = go.Layout(
            title=f"Neighborhood Visualization (Point {center_idx})",
            scene=dict(aspectmode="data"),
        )

        fig = go.Figure(data=data, layout=layout)
        if filename:
            fig.write_html(filename)
            print(f"Visualization saved to {filename}")
        else:
            fig.show()

    def _draw_wireframe_sphere(self, ax, center, radius):
        """Matplotlib 辅助函数：画球"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color="b", alpha=0.1)


def plot_comparison(coords, center_idx, results_dict):
    """
    绘制三个子图进行对比
    results_dict: {'Method Name': (indices, specific_params_dict)}
    """
    fig = plt.figure(figsize=(18, 6))

    # 获取中心点坐标
    center_pos = coords[center_idx]

    # 全局距离矩阵（用于辅助绘图，比如画出被 Dilated 跳过的点）
    dists = np.linalg.norm(coords - center_pos, axis=1)

    for i, (name, (indices, params)) in enumerate(results_dict.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        # 1. 获取当前方法的邻居
        my_neighbors_idx = indices[center_idx]
        my_neighbors_pos = coords[my_neighbors_idx]

        # 2. 绘制背景点 (灰色，很淡)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="gray", s=5, alpha=0.1)

        # 3. 绘制中心点 (红色五角星)
        ax.scatter(
            center_pos[0],
            center_pos[1],
            center_pos[2],
            c="red",
            s=100,
            marker="*",
            label="Center",
            zorder=10,
        )

        # 4. 绘制选中的邻居 (蓝色实心点)
        ax.scatter(
            my_neighbors_pos[:, 0],
            my_neighbors_pos[:, 1],
            my_neighbors_pos[:, 2],
            c="blue",
            s=40,
            label="Selected",
            zorder=9,
        )

        # 5. 绘制连线
        for nb in my_neighbors_pos:
            ax.plot(
                [center_pos[0], nb[0]],
                [center_pos[1], nb[1]],
                [center_pos[2], nb[2]],
                "k--",
                alpha=0.3,
                linewidth=0.8,
            )

        # --- 特殊绘制逻辑 ---

        # 如果是 Ball Query，画球
        if "radius" in params:
            r = params["radius"]
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = center_pos[0] + r * np.cos(u) * np.sin(v)
            y = center_pos[1] + r * np.sin(u) * np.sin(v)
            z = center_pos[2] + r * np.cos(v)
            ax.plot_wireframe(
                x, y, z, color="orange", alpha=0.3, linewidth=0.5, label=f"R={r}"
            )

        # 如果是 Dilated，画出"被跳过"的中间点
        if "dilation" in params:
            d = params["dilation"]
            # 找到最近的 n*d 个点
            nearest_indices = np.argsort(dists)[: len(my_neighbors_idx) * d]
            # 找出在最近范围内，但没被选中的点
            skipped = np.setdiff1d(nearest_indices, my_neighbors_idx)
            skipped_pos = coords[skipped]
            if len(skipped_pos) > 0:
                ax.scatter(
                    skipped_pos[:, 0],
                    skipped_pos[:, 1],
                    skipped_pos[:, 2],
                    c="orange",
                    s=15,
                    alpha=0.8,
                    label="Skipped",
                    marker="x",
                )

        ax.set_title(name)
        # 移除坐标轴刻度以显得更干净
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
