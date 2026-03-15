import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager


OUTPUT_PDF = "assignment_report.pdf"


def configure_fonts() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    selected = None
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            selected = name
            break
        except Exception:
            continue

    if selected:
        plt.rcParams["font.family"] = selected
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False


def new_page(title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait, inches
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.9])
    ax.axis("off")

    ax.text(0.0, 0.98, title, fontsize=22, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.0, 0.94, subtitle, fontsize=11, color="#555", va="top")

    return fig, ax


def add_bullets(ax, y_start: float, lines: list[str], fontsize: int = 12, line_gap: float = 0.048):
    y = y_start
    for line in lines:
        ax.text(0.0, y, f"- {line}", fontsize=fontsize, va="top")
        y -= line_gap
    return y


def find_existing_images() -> list[str]:
    candidates = [
        "pinn_wave_1d_result.png",
        "pinn_wave_1d_loss.png",
        os.path.join("repo2", "pinn_wave_1d_result.png"),
        os.path.join("repo2", "pinn_wave_1d_loss.png"),
        os.path.join("pinn_repo", "pinn_wave_1d_result.png"),
        os.path.join("pinn_repo", "pinn_wave_1d_loss.png"),
    ]
    return [p for p in candidates if os.path.exists(p)]


def exact_solution_figure():
    c = 1.0
    x = np.linspace(0, 1, 200)
    t = np.linspace(0, 1, 200)
    X, T = np.meshgrid(x, t)
    U = np.sin(np.pi * X) * np.cos(c * np.pi * T)

    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(U, extent=[0, 1, 1, 0], aspect="auto", cmap="viridis")
    ax.set_title("解析解示意图  u(x,t)=sin(pi x)cos(pi t)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def build_pdf(path: str) -> None:
    configure_fonts()
    images = find_existing_images()

    with PdfPages(path) as pdf:
        # Page 1: Cover
        fig, ax = new_page("PINN 项目作业报告", "Physics-Informed Neural Networks（物理信息神经网络）")
        ax.text(0.0, 0.84, "题目：1D 双曲型方程（波动方程）求解", fontsize=14, fontweight="bold")
        ax.text(0.0, 0.78, "方程：u_tt = c^2 u_xx,  x∈[0,1], t∈[0,1]", fontsize=13)
        ax.text(0.0, 0.74, "条件：u(x,0)=sin(pi x),  u_t(x,0)=0,  u(0,t)=u(1,t)=0", fontsize=13)

        ax.text(0.0, 0.64, "报告摘要", fontsize=15, fontweight="bold")
        y = add_bullets(
            ax,
            0.60,
            [
                "使用 PyTorch 搭建 PINN，网络输入为 (x,t)，输出为 u(x,t)。",
                "通过自动微分构造 PDE 残差，将物理约束并入损失函数。",
                "训练策略采用 Adam 预训练 + L-BFGS 精修。",
                "输出并分析预测场图、绝对误差图与损失曲线。",
            ],
            fontsize=12,
        )
        ax.text(0.0, y - 0.02, f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=10, color="#666")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Method
        fig, ax = new_page("1. 方法原理", "PINN = 数据约束 + 物理方程约束")
        ax.text(0.0, 0.86, "神经网络近似", fontsize=14, fontweight="bold")
        ax.text(0.02, 0.82, "u_theta(x,t) ≈ u(x,t)", fontsize=13)

        ax.text(0.0, 0.75, "损失函数构成", fontsize=14, fontweight="bold")
        lines = [
            "PDE 残差：L_pde = MSE(u_tt - c^2 u_xx)",
            "初值位移：L_ic_u = MSE(u(x,0) - sin(pi x))",
            "初值速度：L_ic_ut = MSE(u_t(x,0) - 0)",
            "边界条件：L_bc = MSE(u(0,t)-0) + MSE(u(1,t)-0)",
            "总损失：L = w1*L_pde + w2*L_ic_u + w3*L_ic_ut + w4*L_bc",
        ]
        add_bullets(ax, 0.71, lines, fontsize=12)

        ax.text(0.0, 0.41, "训练采样", fontsize=14, fontweight="bold")
        add_bullets(
            ax,
            0.37,
            [
                "域内采样点（collocation points）用于约束 PDE 残差。",
                "初值点用于约束 t=0 时刻的位移和速度。",
                "边界点用于约束 x=0 与 x=1 处边界条件。",
            ],
            fontsize=12,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Implementation
        fig, ax = new_page("2. 实现与流程", "基于 PyTorch 的可复现实验流程")
        add_bullets(
            ax,
            0.86,
            [
                "网络结构：多层感知机（MLP），激活函数使用 tanh。",
                "自动微分：通过 autograd 计算 u_t, u_tt, u_x, u_xx。",
                "优化策略：先 Adam（快速下降），再 L-BFGS（高精修正）。",
                "评估方式：与解析解对比，观察绝对误差与相对误差。",
            ],
            fontsize=12,
        )

        ax.text(0.0, 0.58, "推荐运行命令", fontsize=14, fontweight="bold")
        ax.text(0.02, 0.54, "python pinn_wave_1d.py", fontsize=12, family="monospace")
        ax.text(0.02, 0.50, "python -c \"import torch, matplotlib; print(torch.__version__)\"", fontsize=11, family="monospace")

        ax.text(0.0, 0.42, "工程说明", fontsize=14, fontweight="bold")
        add_bullets(
            ax,
            0.38,
            [
                "本报告根据项目目标与已完成实验流程自动生成。",
                "若目录内存在实验图像，将自动插入报告中。",
                "若图像缺失，报告将使用理论解示意图与文字分析替代。",
            ],
            fontsize=12,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Results
        fig, ax = new_page("3. 结果分析", "模型输出与误差表现")
        if images:
            ax.text(0.0, 0.88, "检测到项目结果图像，已插入如下：", fontsize=12)
            first_img = images[0]
            second_img = images[1] if len(images) > 1 else None

            # place first image
            img1 = plt.imread(first_img)
            iax1 = fig.add_axes([0.10, 0.48, 0.80, 0.34])
            iax1.imshow(img1)
            iax1.axis("off")
            iax1.set_title(os.path.basename(first_img), fontsize=10)

            if second_img:
                img2 = plt.imread(second_img)
                iax2 = fig.add_axes([0.10, 0.12, 0.80, 0.28])
                iax2.imshow(img2)
                iax2.axis("off")
                iax2.set_title(os.path.basename(second_img), fontsize=10)
        else:
            ax.text(0.0, 0.88, "未检测到本地实验图片（pinn_wave_1d_result.png / pinn_wave_1d_loss.png）。", fontsize=12, color="#b00020")
            ax.text(0.0, 0.83, "已使用理论解析解示意图替代，便于作业说明。", fontsize=12)

            demo_fig = exact_solution_figure()
            # render demo figure as image and place on page
            demo_fig.canvas.draw()
            w, h = demo_fig.canvas.get_width_height()
            buf = np.frombuffer(demo_fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            plt.close(demo_fig)

            iax = fig.add_axes([0.12, 0.40, 0.76, 0.38])
            iax.imshow(buf)
            iax.axis("off")

            add_bullets(
                ax,
                0.33,
                [
                    "理论解随时间呈余弦振荡，空间维保持正弦模态。",
                    "PINN 的目标是同时满足该时空演化规律与边界/初值约束。",
                    "实际训练中应重点关注：误差热图、切片重合度、损失收敛趋势。",
                ],
                fontsize=12,
                line_gap=0.055,
            )

        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Conclusion
        fig, ax = new_page("4. 结论与改进方向", "作业总结")
        add_bullets(
            ax,
            0.86,
            [
                "PINN 适合‘数据较少但物理规律明确’的问题。",
                "对 1D 波动方程，PINN 可通过自动微分直接学习 PDE 解。",
                "组合优化器（Adam + L-BFGS）通常能提高收敛稳定性。",
                "损失项权重平衡是训练质量的关键因素之一。",
            ],
            fontsize=12,
            line_gap=0.055,
        )

        ax.text(0.0, 0.53, "可继续完善", fontsize=14, fontweight="bold")
        add_bullets(
            ax,
            0.49,
            [
                "加入自适应损失加权与误差驱动重采样（RAR）。",
                "引入 Sobol / LHS 采样提升点集覆盖质量。",
                "扩展到更复杂双曲型方程与参数反演任务。",
            ],
            fontsize=12,
            line_gap=0.055,
        )

        ax.text(0.0, 0.30, "报告文件", fontsize=13, fontweight="bold")
        ax.text(0.02, 0.26, f"{OUTPUT_PDF}", fontsize=12, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    build_pdf(OUTPUT_PDF)
    print(f"Generated: {os.path.abspath(OUTPUT_PDF)}")
