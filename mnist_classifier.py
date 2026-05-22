import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# On essaye de voir si l'on peut utiliser la carte graphique plutot que le processeur
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Utilisation de l'appareil : {device}")

# --- 2. PRÉPARATION DES DONNÉES ---
# On convertit les images en Tensors et on les normalise
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Moyenne et écart-type standards pour MNIST
])

# Téléchargement de 60 000 images d'entraînement et de 10 000 image de test
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# On coupe les données en paquet
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


# --- 3. ARCHITECTURE DU RÉSEAU DE NEURONES ---
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        # Une image MNIST fait 28x28 pixels = 784 entrées
        # Couche 1 : 784 pixels -> 128 neurones cachés
        self.fc1 = nn.Linear(28 * 28, 128)
        # Couche de sécurité (ReLU) pour introduire de la non-linéarité
        self.relu = nn.ReLU()
        # Couche 2 : 128 neurones -> 10 sorties (une pour chaque chiffre de 0 à 9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # On aplatit l'image de (28, 28) à un vecteur plat de 784 éléments
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNeuralNet().to(device)


# --- 4. ENTRAÎNEMENT DE L'IA ---
criterion = nn.CrossEntropyLoss() # La fonction qui calcule l'erreur du modèle
optimizer = optim.SGD(model.parameters(), lr=0.01) # L'algorithme qui ajuste les neurones (Descente de gradient)

print("\nDébut de l'entraînement...")
model.train()

# On fait passer toutes les données 3 fois (3 époques)
for epoch in range(1, 4):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()   # Réinitialise les calculs précédents
        output = model(data)    # L'IA fait sa prédiction
        loss = criterion(output, target) # On calcule son erreur
        loss.backward()         # On calcule comment corriger les neurones
        optimizer.step()        # On applique la correction

        if batch_idx % 200 == 0:
            print(f"Époque {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] - Erreur : {loss.item():.4f}")


# --- 5. ÉVALUATION DU MODÈLE ---
model.eval()
correct = 0

with torch.no_grad(): # Pas besoin de calculer les corrections pendant le test
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) # On prend le chiffre avec le score le plus haut
        correct += pred.eq(target.view_as(pred)).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f"\nPrécision finale sur les images de test : {accuracy:.2f}%")

# --- 6. VISUALISATION D'UNE PRÉDICTION ---
# On récupère un paquet d'images de test
data_iter = iter(test_loader)
images, labels = next(data_iter)

# On prend la première image de ce paquet
img = images[0]
true_label = labels[0].item()

# L'IA fait sa prédiction sur cette image unique
model.eval()
with torch.no_grad():
    # On ajoute une dimension pour simuler un "batch" de 1 image et on l'envoie sur le CPU/GPU
    output = model(img.unsqueeze(0).to(device))
    prediction = output.argmax(dim=1).item()

# Affichage visuel avec Matplotlib
# On désactive la normalisation juste pour l'affichage écran
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Vrai chiffre : {true_label} | Prédiction IA : {prediction}")
plt.axis('off') # Cache les axes X et Y
plt.show()