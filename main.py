import torch

device = "cuda" if torch.cuda.is_available() else "cpu"  # 장치 선택

# 입력
X = torch.Tensor([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]]).to(device)

# 정답
Y = torch.Tensor([[0],
                  [1],
                  [1],
                  [0]]).to(device)


# XOR 모델
class XOR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(3, 5)
        self.layer2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


model = XOR().to(device)

criterion = torch.nn.BCELoss().to(device)  # Binary Cross Entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)  # SGD 사용, 학습률 0.02
for t in range(10000):
    y_pred = model(X)

    loss = criterion(y_pred, Y)  # loss 계산
    if t % 1000 == 999:
        print(f'epoch: {t + 1}/{10000}, loss: {loss.item()}')

    optimizer.zero_grad()  # 초기화
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 갱신

with torch.no_grad():
    print(f'출력: {model(X)}')
