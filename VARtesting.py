import torch
from VARtraining import model1, model2, test_loader  
from torch import nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load models

# Load the saved model parameters
model1.load_state_dict(torch.load("C:\\Users\\김해창\\Desktop\\VAR\\pth\\model1.pth"))
model2.load_state_dict(torch.load("C:\\Users\\김해창\\Desktop\\VAR\\pth\\model2.pth"))

model1.to(device)
model2.to(device)

model1.eval()
model2.eval()  # Set model to evaluation mode

criterion = nn.CrossEntropyLoss()

def evaluate_model(model,test_loader):
    correct = 0  #테스트에서 맞춘 답 수, 불리언 텐서로 축적될 것임
    total = 0   #테스트 데이터셋에서 전체 샘플 수, for문 안에서 lables.size(0)으로 배치로 나눠진 매 반복의 배치의 크기가 축적될 것임
    examples = []
    num_examples = 16
    
    with torch.no_grad():   #이제부터 모든 연산은 기울기를 계산하지 않음,  평가 시에는 보통 이걸 쓰는 듯
        for images, labels in test_loader:
            images,labels = images.to(device), labels.to(device)
            outputs = model(images) # forward() 호출됨
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if len(examples) < num_examples:
                for i in range(len(labels)):
                    if len(examples) < num_examples:
                        examples.append((images[i], labels[i], predicted[i]))
                    else:
                        break
    accuracy = 100 * correct / total
    print(f'Test Accuracy of the model on the {total} test images: {accuracy}%')
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, (image, label, prediction) in enumerate(examples):
        image = image.squeeze().cpu().numpy() * 0.5 + 0.5
        ax = axes[i // 4, i % 4]
        ax.imshow(image, cmap='gray')
        ax.set_title(f'True: {label.item()}\nPred: {prediction.item()}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

evaluate_model(model1, test_loader)

# Evaluate model_2
evaluate_model(model2, test_loader)