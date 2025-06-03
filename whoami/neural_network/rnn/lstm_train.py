#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/16 14:25
@Author  : weiyutao
@File    : lstm_train.py
"""
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from whoami.neural_network.rnn.model import LSTM

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = dsets.MNIST(root='/work/ai/public_data', 
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='/work/ai/public_data', 
                           train=False, 
                           transform=transforms.ToTensor())
    
    batch_size = 300
    n_iters = 30000
    num_epochs = int(n_iters / (len(train_dataset) / batch_size))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
    
    # 设置随机种子
    torch.manual_seed(42)
    output_dim = 10
    feature_size = 28
    hidden_size = 128
    bidirectional = False
    # 分类器动态适应单向或者双向的输出维度
    classifier_output_dim = 2*hidden_size if bidirectional else hidden_size
    custom_lstm = LSTM(feature_size, hidden_size, 1, dropout=0.2, bidirectional=bidirectional).to(device)
    # custom_lstm = nn.LSTM(feature_size, hidden_size, 3, dropout=0.2, bidirectional=bidirectional).to(device)
    classifier = nn.Linear(classifier_output_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam([
        {'params': custom_lstm.parameters()},
        {'params': classifier.parameters()}
    ], lr=0.001)
    
    # 评估函数 - 适用于LSTM
    def evaluate(model, classifier, data_loader):
        model.eval()  # 设置为评估模式
        classifier.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():  # 禁用梯度计算
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                images = images.squeeze(1)
                
                # 前向传播 - LSTM返回(outputs, (h_n, c_n))
                outputs, (h_n, c_n) = model(images, None)
                
                # 对最后一个时间步的输出进行分类
                logits = classifier(outputs[:, -1, :])
                
                # 计算准确率
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    # 记录最佳模型和性能
    train_loss = 0.0
    best_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        custom_lstm.train()
        classifier.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Zero gradients at the start of each batch
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            images = images.squeeze(1)  # 去掉通道维度 [batch, 1, 28, 28] -> [batch, 28, 28]
            
            # 前向传播 - LSTM返回(outputs, (h_n, c_n))
            outputs, (h_n, c_n) = custom_lstm(images, None)
            
            # 使用最后一个时间步的输出进行分类
            logits = classifier(outputs[:, -1, :])
            
            loss = criterion(logits, labels)
            train_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(
                list(custom_lstm.parameters()) + list(classifier.parameters()), 
                max_norm=1.0
            )
            
            # 更新权重
            optimizer.step()
            
            if (batch_idx+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 计算训练集上的平均损失
        train_loss /= len(train_loader)
        
        # 评估阶段
        train_accuracy = evaluate(custom_lstm, classifier, train_loader)
        test_accuracy = evaluate(custom_lstm, classifier, test_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%')
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = {
                'lstm': custom_lstm.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'test_accuracy': test_accuracy
            }
            # 保存模型
            torch.save(best_model_state, 'best_lstm_mnist_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.2f}%')
    
    # 训练完成后，加载最佳模型进行最终评估
    if best_model_state:
        print("\nLoading best model for final evaluation...")
        custom_lstm.load_state_dict(best_model_state['lstm'])
        classifier.load_state_dict(best_model_state['classifier'])
        
        final_test_accuracy = evaluate(custom_lstm, classifier, test_loader)
        print(f'Final Test Accuracy with Best Model: {final_test_accuracy:.2f}%')
        print(f'Best model was from epoch {best_model_state["epoch"] + 1}')