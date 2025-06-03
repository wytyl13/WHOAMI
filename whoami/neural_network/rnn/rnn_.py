import torch
from torch import nn

from whoami.neural_network.rnn.model import RNN



if __name__ == '__main__':

    # 设置随机种子
    torch.manual_seed(42)

    # 创建两个 RNN
    custom_rnn = RNN(10, 20, 2, nonlinearity='relu')
    pytorch_rnn = nn.RNN(10, 20, 2, nonlinearity='relu')
    print(custom_rnn.nonlinearity)
    print(pytorch_rnn.nonlinearity)
    # 同步权重：从 PyTorch RNN 复制权重到自定义 RNN
    with torch.no_grad():
        for name, param in pytorch_rnn.named_parameters():
            if hasattr(custom_rnn, name):
                custom_param = getattr(custom_rnn, name)
                custom_param.data.copy_(param.data)

    # 创建相同的输入
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)

    # 设置为评估模式（关闭 dropout）
    custom_rnn.eval()
    pytorch_rnn.eval()

    # 前向传播
    output_custom, hn_custom = custom_rnn(input, h0)
    output_pytorch, hn_pytorch = pytorch_rnn(input, h0)

    # 比较结果
    print("输出形状:")
    print(f"自定义 RNN: {output_custom.shape}")
    print(f"PyTorch RNN: {output_pytorch.shape}")

    print("\n隐藏状态形状:")
    print(f"自定义 RNN: {hn_custom.shape}")
    print(f"PyTorch RNN: {hn_pytorch.shape}")

    # 计算差异
    output_diff = torch.abs(output_custom - output_pytorch).max().item()
    hidden_diff = torch.abs(hn_custom - hn_pytorch).max().item()

    print(f"\n最大输出差异: {output_diff:.2e}")
    print(f"最大隐藏状态差异: {hidden_diff:.2e}")

    # 检查是否足够接近
    if output_diff < 1e-6 and hidden_diff < 1e-6:
        print("\n✓ 两个 RNN 的输出基本相同！")
    else:
        print("\n✗ 两个 RNN 的输出有较大差异。")
        
    # 显示输出的最后一维大小
    print(f"\n输出最后一维大小: {output_custom.size(2)}")