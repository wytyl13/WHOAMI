#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/15 16:13
@Author  : weiyutao
@File    : train.py
"""
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from whoami.neural_network.rnn.model import RNN

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
    n_iters = 6000
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
    bidirectional = True
    custom_rnn = RNN(feature_size, hidden_size, 3, nonlinearity='relu', bidirectional=bidirectional).to(device)
    # custom_rnn = nn.RNN(feature_size, hidden_size, 3, nonlinearity='relu').to(device)
    
    # 分类器动态适应单向或者双向的输出维度
    classifier_output_dim = 2*hidden_size if bidirectional else hidden_size
    classifier = nn.Linear(classifier_output_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam([
        {'params': custom_rnn.parameters()},
        {'params': classifier.parameters()}
    ], lr=0.001)
    
    
    # 评估函数
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
                
                # 前向传播
                outputs, _ = model(images, None)
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
        custom_rnn.train()
        classifier.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Zero gradients at the start of each batch
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            images = images.squeeze(1)
            outputs, hidden = custom_rnn.forward(images, None)
            logits = classifier(outputs[:, -1, :])
            loss = criterion(logits, labels)
            train_loss += loss.item()
            # Backward pass
            loss.backward()
            # Add gradient clipping (recommended to prevent explosion)
            torch.nn.utils.clip_grad_norm_(list(custom_rnn.parameters()) + list(classifier.parameters()), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            if (batch_idx+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
        # 计算训练集上的平均损失
        train_loss /= len(train_loader)
        # 评估阶段
        train_accuracy = evaluate(custom_rnn, classifier, train_loader)
        test_accuracy = evaluate(custom_rnn, classifier, test_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%')
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = {
                'rnn': custom_rnn.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'test_accuracy': test_accuracy
            }
            # 保存模型
            torch.save(best_model_state, 'best_rnn_mnist_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.2f}%')
    # 训练完成后，加载最佳模型进行最终评估
    if best_model_state:
        print("\nLoading best model for final evaluation...")
        custom_rnn.load_state_dict(best_model_state['rnn'])
        classifier.load_state_dict(best_model_state['classifier'])
        
        final_test_accuracy = evaluate(custom_rnn, classifier, test_loader)
        print(f'Final Test Accuracy with Best Model: {final_test_accuracy:.2f}%')
        print(f'Best model was from epoch {best_model_state["epoch"] + 1}')